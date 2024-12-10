'''
Evaluate the linear probe performance on different checkpoints with vocal modality support.
'''

import logging
import os
import random
from datetime import datetime
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import glob

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from clap_module import create_model, trace_model
from training.data import get_data
from training.params import parse_args
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.scheduler import cosine_lr
from training.lp_main import config_lp_optimizer
from training.lp_train import train_one_epoch, evaluate
from clap_module.utils import get_tar_path_from_dataset_name, dataset_split
from clap_module.utils import load_p, load_class_label
from clap_module.linear_probe import LinearProbe

def maintain_ckpts(args, startidx, all_idx_len):
    for i in reversed(range(startidx, all_idx_len)):
        if os.path.exists(os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt")):
            os.rename(
                os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt"),
                os.path.join(args.checkpoint_path, f"epoch_top_{i+1}.pt"),
            )
    if os.path.exists(
            os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt")
    ):
        os.remove(os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt"))
    return

def update_top_k_performance(
        new_metrics_inputs, current_top_k_ckpt_metrics, args, ckpt, bignumbetter=True, pretrain_epoch=0
):
    """
    Record the top-k performance of the current epoch.
    current_top_k_metrics is a dictionary of the form: {1: top_1_ckpt_measure, 2: top_2_ckpt_measure, ...}
    """
    if isinstance(new_metrics_inputs, (list, tuple)):
        new_metrics_inputs = np.mean(new_metrics_inputs)
        return update_top_k_performance(
            new_metrics_inputs,
            current_top_k_ckpt_metrics,
            args=args,
            ckpt=ckpt,
            bignumbetter=bignumbetter,
            pretrain_epoch=pretrain_epoch
        )
    elif isinstance(new_metrics_inputs, dict):
        new_metrics_inputs = np.mean(list(new_metrics_inputs.values()))
        return update_top_k_performance(
            new_metrics_inputs,
            current_top_k_ckpt_metrics,
            args=args,
            ckpt=ckpt,
            bignumbetter=bignumbetter,
            pretrain_epoch=pretrain_epoch
        )
    elif isinstance(new_metrics_inputs, (float, int)):
        update_flag = {k: False for k in current_top_k_ckpt_metrics.keys()}
        sorted_keys = sorted(current_top_k_ckpt_metrics.keys())
        sorted_values = sorted(
            current_top_k_ckpt_metrics.values(), reverse=bignumbetter
        )
        sorted_values_ = copy.deepcopy(sorted_values)
        sorted_values.append(new_metrics_inputs)
        sorted_values = sorted(sorted_values, reverse=bignumbetter)
        sorted_values = sorted_values[:-1]

        if sorted_values == sorted_values_:
            return current_top_k_ckpt_metrics, new_metrics_inputs
        else:
            for i in range(len(sorted_keys)):
                if current_top_k_ckpt_metrics[sorted_keys[i]] != sorted_values[i]:
                    current_top_k_ckpt_metrics[sorted_keys[i]] = sorted_values[i]
                    update_flag[sorted_keys[i]] = True
            for i in range(len(update_flag)):
                if update_flag[i]:
                    maintain_ckpts(args, i, len(sorted_keys))
                    torch.save(
                        ckpt,
                        os.path.join(args.checkpoint_path, f"pretrain_epoch_{pretrain_epoch}_lp_epoch_top_{i}.pt"),
                    )
                    break
            return current_top_k_ckpt_metrics, new_metrics_inputs

def is_pretrained_params(n):
    return (
            n.startswith("clap_model.transformer")
            or n in ["clap_model.positional_embedding", "clap_model.text_projection"]
            or n.startswith("clap_model.token_embedding")
            or n.startswith("clap_model.ln_final")
            or n.startswith("clap_model.logit_scale_t")
    )

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main():
    args = parse_args()
    # Sanitize model names
    args.amodel = args.amodel.replace("/", "-")
    args.tmodel = args.tmodel.replace("/", "-")
    args.vmodel = args.vmodel.replace("/", "-")

    pretrained_ckpts = sorted(glob.glob(os.path.join(args.pretrained, "*.pt")), key=os.path.getmtime)

    if args.name is None:
        args.name = "-".join(
            [
                datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
                "linear_probe",
                f"model_{args.amodel}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    # Initialize distributed environment
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if args.remotedata and is_master(args):
        for dataset_name in args.datasetnames:
            for split in dataset_split[dataset_name]:
                if not os.path.exists(f"./json_files/{dataset_name}/{split}"):
                    os.makedirs(f"./json_files/{dataset_name}/{split}")
                os.system(
                    f"aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json ./json_files/{dataset_name}/{split}/sizes.json"
                )
    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)

        # Avoid log dir with the same name
        postfix = 0
        while os.path.exists(args.log_path):
            postfix += 1
            log_base_path_new = log_base_path + '-' + str(postfix)
            os.makedirs(log_base_path_new, exist_ok=True)
            log_filename = f"out-{args.rank}" if args.log_local else "out.log"
            args.log_path = os.path.join(log_base_path_new, log_filename)

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Initialize device
    device = init_distributed_device(args)

    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(args.logs, args.name, "tensorboard")
            if args.tensorboard
            else ""
        )
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""
        args.checkpoint_path = ""

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ["amp", "fp16", "fp32"]
    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for training."
        )

    if args.horovod:
        logging.info(
            f"Running in horovod mode with multiple processes/nodes. Device: {args.device}. "
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}. "
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    logging.info(f'OpenAI cache dir: {os.path.expanduser(args.openai_model_cache_dir)}')

    # Determine if this worker should save logs and checkpoints
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        wandb.init(
            project="clvap_linear_probe",
            notes=args.wandb_notes,
            name=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        logging.debug("Finished loading wandb.")

    for idx, f in enumerate(pretrained_ckpts):
        logging.info(f"Using pretrained checkpoint {f}")
        args.pretrained = f
        ckpt = torch.load(f, map_location='cpu')
        pretrain_epoch = ckpt.get('epoch', 0)

        # Run linear probe main function
        best_metrics = lp_main(args, device, writer, pretrain_epoch, idx)

        if args.wandb and is_master(args):
            assert wandb is not None, "Please install wandb."
            for name, val in best_metrics.items():
                wandb.log({f"val/summary/{name}": val, "epoch": pretrain_epoch})

    if args.wandb and is_master(args):
        wandb.finish()

def update_metric(best_metric, new_metric):
    for key in new_metric:
        if key not in best_metric:
            best_metric[key] = new_metric[key]
        else:
            best_metric[key] = max(best_metric[key], new_metric[key])
    return best_metric

def lp_main(args, device, writer, pretrain_epoch, idx):
    random_seed(args.seed, rank=args.rank)

    args.class_index_dict = load_class_label(args.class_label_path)

    # Create CLVAP model
    clap_model, clap_model_cfg = create_model(
        amodel_name=args.amodel,
        tmodel_name=args.tmodel,
        vmodel_name=args.vmodel,
        pretrained=args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
        skip_params=False,
        enable_fusion=args.enable_fusion,
        fusion_type=args.fusion_type
    )

    args.lp_out_ch = len(args.class_index_dict)
    # Linear Probe
    if idx == 0:
        logging.info(f"Linear probe using MLP: {args.lp_mlp}")
        logging.info(f"Linear probe freeze: {args.lp_freeze}")
        logging.info(f"Linear probe activation layer: {args.lp_act}")
        logging.info(f"Linear probe output channels: {args.lp_out_ch}")
        logging.info(f"Linear probe learning rate: {args.lp_lr}")
        logging.info(f"Linear probe loss function: {args.lp_loss}")
        logging.info(f"Linear probe metrics: {args.lp_metrics}")

    model = LinearProbe(
        clap_model,
        mlp=args.lp_mlp,
        freeze=args.lp_freeze,
        in_ch=512,
        out_ch=args.lp_out_ch,
        act=args.lp_act
    )
    model = model.to(device)

    if args.horovod:
        with torch.no_grad():
            for param in model.parameters():
                param.set_(param.contiguous())

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if is_master(args) and idx == 0:
        logging.info("Linear Probe CLVAP Model:")
        logging.info(f"{str(clap_model)}")
        logging.info("Parameters:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True, **ddp_args
        )

    data = get_data(args, clap_model_cfg)
    assert len(data), "At least one train or eval dataset must be specified."
    if args.trace:
        assert "train" not in data, "Cannot train with traced model"

    optimizer, scheduler, freeze_parameters = config_lp_optimizer(model, data, args)

    scaler = GradScaler() if args.precision == "amp" else None

    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if "epoch" in checkpoint:
                # Resuming a train checkpoint with epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
                    sd = {k[len("module."):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                logging.info(
                    f"=> Resuming checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            else:
                # Loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(
                    f"=> Loaded checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            if args.freeze_text:
                logging.info("Freeze Text!")
                for k in freeze_parameters:
                    k.requires_grad = False
        else:
            logging.info("=> No checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    if args.wandb and is_master(args):
        if "train" in data:
            args.train_sz = data["train"].dataloader.num_samples
        if "val" in data:
            args.val_sz = data["val"].dataloader.num_samples
        if args.debug:
            wandb.watch(model, log="all")
        if idx == 0:
            wandb.save(params_file)

    best_metrics = {}

    if "train" not in data:
        metric = evaluate(model, data, start_epoch, args, writer, extra_suffix="_pe@" + str(pretrain_epoch))
        if is_master(args):
            best_metrics = update_metric(best_metrics, metric)
        return best_metrics
    elif start_epoch == 0 and "val" in data and not args.no_eval:
        metric = evaluate(model, data, 0, args, writer, extra_suffix="_pe@" + str(pretrain_epoch))
        if is_master(args):
            best_metrics = update_metric(best_metrics, metric)
    if args.save_top_performance:
        current_top_k_ckpt_metrics = {
            i: 0 for i in range(args.save_top_performance)
        }  # Initialize the top-k metric for ckpts to 0

    for epoch in range(start_epoch, args.epochs):
        # Freeze the text parameters after args.freeze_text_after epoch
        if epoch == args.freeze_text_after:
            logging.info("Text pretrained parameters are frozen since this epoch.")
            for k in freeze_parameters:
                k.requires_grad = False
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer, extra_suffix="_pe@" + str(pretrain_epoch))
        completed_epoch = epoch + 1

        if any(v in data for v in ("val", "imagenet-val", "imagenet-v2")) and not args.no_eval:
            metric = evaluate(model, data, completed_epoch, args, writer, extra_suffix="_pe@" + str(pretrain_epoch))
            if is_master(args):
                best_metrics = update_metric(best_metrics, metric)
            if args.save_top_performance:
                top_k_dataset = args.top_k_checkpoint_select_dataset
                top_k_metric = args.top_k_checkpoint_select_metric
                filtered_metrics = [
                    v
                    for k, v in metric.items()
                    if top_k_metric in k and top_k_dataset in k
                ]
        # Saving checkpoints
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "pretrain_epoch": pretrain_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"pretrain_epoch_{pretrain_epoch}_lp_epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"pretrain_epoch_{pretrain_epoch}_lp_epoch_latest.pt"),
                )
            if args.save_top_performance and not args.no_eval:
                update_top_k_performance(
                    filtered_metrics,
                    current_top_k_ckpt_metrics,
                    args,
                    checkpoint_dict,
                    bignumbetter=True,
                    pretrain_epoch=pretrain_epoch
                )
    del clap_model
    return best_metrics

def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
    )
    print("Done copying code.")
    return 1

if __name__ == "__main__":
    main()
