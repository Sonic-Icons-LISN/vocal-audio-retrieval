"""
Evaluate the retrieval performance on different checkpoints for the CLVAP model.
"""

import os
import glob
import random
import numpy as np
import logging
import torch
import torch.backends.cudnn as cudnn

try:
    import wandb
except ImportError:
    wandb = None

from clap_module import create_model
from clap_module.training.logger import setup_logging
from clap_module.training.data import get_data
from clap_module.training.train import evaluate
from clap_module.utils import get_tar_path_from_dataset_name, dataset_split
from clap_module.training.params import parse_args


def find_params_value(file, key):
    # Find the value of a parameter in the params_file
    with open(file, 'r') as f:
        for line in f:
            if key + ': ' in line:
                return line.split(': ')[1].strip()
    return None


if __name__ == '__main__':
    # Set seeds for reproducibility
    seed = 3407
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = False

    args = parse_args()

    # Determine log directory based on pretrained model path
    if os.path.isdir(args.pretrained):
        log_dir = os.path.dirname(args.pretrained)
    else:
        log_dir = os.path.dirname(os.path.dirname(args.pretrained))

    # Setup logging
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_path = os.path.join(log_dir, 'out.log')
    setup_logging(log_path, args.log_level)
    params_file = os.path.join(log_dir, 'params.txt')

    # Load model configurations from params file
    amodel = find_params_value(params_file, 'amodel')
    tmodel = find_params_value(params_file, 'tmodel')
    vmodel = find_params_value(params_file, 'vmodel')  # Added for vocal model

    if amodel is None or tmodel is None or vmodel is None:
        raise ValueError('Model type not found in params file')

    # Set up default values for args
    args.parallel_eval = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.val_frequency = 1
    args.epochs = 1
    args.precision = 'fp32'
    args.save_logs = True
    args.wandb = True
    args.class_index_dict = None
    args.enable_fusion = True  # Enable fusion for text and vocal modalities
    args.fusion_type = 'default'  # Set the appropriate fusion type if needed

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Handle remote data if necessary
    if args.remotedata:
        for dataset_name in args.datasetnames:
            for split in dataset_split[dataset_name]:
                split_path = f"./json_files/{dataset_name}/{split}"
                if not os.path.exists(split_path):
                    os.makedirs(split_path)
                os.system(
                    f"aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json {split_path}/sizes.json"
                )

    # Prepare data paths
    if args.datasetinfos is None:
        args.datasetinfos = ["train", "unbalanced_train", "balanced_train"]
    if args.dataset_type == "webdataset":
        args.train_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            args.datasetinfos,
            islocal=not args.remotedata,
            proportion=args.dataset_proportion,
            dataset_path=args.datasetpath,
        )
        args.val_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            ["valid", "test", "eval"],
            islocal=not args.remotedata,
            proportion=1,
            dataset_path=args.datasetpath,
        )

    # Create CLVAP model with vocal modality
    model, model_cfg = create_model(
        amodel_name=amodel,
        tmodel_name=tmodel,
        vmodel_name=vmodel,  # Include vocal model
        pretrained='openai',  # Change if using different pretrained weights
        precision=args.precision,
        device=device,
        jit=False,
        force_quick_gelu=False,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
        skip_params=False,
        enable_fusion=args.enable_fusion,
        fusion_type=args.fusion_type
    )

    # Load data
    data = get_data(args, model_cfg=model_cfg)

    writer = None  # Initialize tensorboard writer if needed

    # Setup Weights & Biases logging
    if args.wandb:
        if wandb is not None:
            logging.debug("Starting wandb.")
            if args.wandb_id is not None:
                wandb.init(
                    project="clvap",
                    id=args.wandb_id,
                    resume=True
                )
            else:
                wandb.init(
                    project="clvap",
                    notes=args.wandb_notes,
                    name=args.wandb_notes,
                    tags=[],
                    config=vars(args),
                )
            logging.debug("Finished loading wandb.")
        else:
            logging.warning("wandb not installed, proceeding without wandb logging.")
            args.wandb = False

    # Load checkpoints
    if os.path.isdir(args.pretrained):
        all_model_checkpoints = sorted(
            glob.glob(os.path.join(args.pretrained, '*.pt')),
            key=os.path.getmtime
        )
    else:
        all_model_checkpoints = [args.pretrained]

    # Evaluate each checkpoint
    for model_path in all_model_checkpoints:
        args.checkpoint_path = os.path.dirname(model_path)
        logging.info(f"Evaluating checkpoint: {model_path}")

        # Recreate the model for each checkpoint
        model, model_cfg = create_model(
            amodel_name=amodel,
            tmodel_name=tmodel,
            vmodel_name=vmodel,  # Include vocal model
            pretrained='openai',  # Change if using different pretrained weights
            precision=args.precision,
            device=device,
            jit=False,
            force_quick_gelu=False,
            openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
            skip_params=False,
            enable_fusion=args.enable_fusion,
            fusion_type=args.fusion_type
        )

        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if "epoch" in checkpoint:
            # Resuming from a training checkpoint
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            if next(iter(state_dict.items()))[0].startswith("module"):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            logging.info(f"=> Resumed checkpoint '{model_path}' (epoch {start_epoch})")
        else:
            # Loading model weights directly
            model.load_state_dict(checkpoint)
            start_epoch = 0
            logging.info(f"=> Loaded checkpoint '{model_path}'")

        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Evaluate the model
        evaluate(model, data, start_epoch, args, writer)

    # Finish wandb logging
    if args.wandb and wandb is not None:
        wandb.finish()
