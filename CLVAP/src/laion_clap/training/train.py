import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from clap_module import ClipLoss, gather_features
from .distributed import is_master


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def train_one_epoch(
        model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None
):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    model.train()
    loss_fn = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        mlp_loss=args.clap_mlploss,
        weight_loss_kappa=args.kappa,
    )

    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # For toy dataset
    if args.dataset_type == "toy":
        dataloader.dataset.generate_queue()

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        if isinstance(scheduler, dict):
            for s in scheduler.values():
                s(step)
        else:
            scheduler(step)

        # Extract data from batch
        audios = batch["waveform"]
        texts = batch["text"]
        vocals = batch["vocal_waveform"]

        data_time_m.update(time.time() - end)
        if isinstance(optimizer, dict):
            for o_ in optimizer.values():
                o_.zero_grad()
        else:
            optimizer.zero_grad()

        with autocast():
            (
                audio_features,
                text_features,
                vocal_features,
                audio_features_mlp,
                text_features_mlp,
                vocal_features_mlp,
                logit_scale_a,
                logit_scale_t,
                logit_scale_v,
            ) = model(audios, texts, vocals, device)

            if args.clap_mlploss:
                total_loss = loss_fn(
                    audio_features=audio_features,
                    text_features=text_features,
                    vocal_features=vocal_features,
                    logit_scale_a=logit_scale_a,
                    logit_scale_t=logit_scale_t,
                    logit_scale_v=logit_scale_v,
                    audio_features_mlp=audio_features_mlp,
                    text_features_mlp=text_features_mlp,
                    vocal_features_mlp=vocal_features_mlp,
                )
            else:
                total_loss = loss_fn(
                    audio_features=audio_features,
                    text_features=text_features,
                    vocal_features=vocal_features,
                    logit_scale_a=logit_scale_a,
                    logit_scale_v=logit_scale_v,
                )

        if isinstance(optimizer, dict):
            if scaler is not None:
                scaler.scale(total_loss).backward()
                for o_ in optimizer.values():
                    if args.horovod:
                        o_.synchronize()
                        scaler.unscale_(o_)
                        with o_.skip_synchronize():
                            scaler.step(o_)
                    else:
                        scaler.step(o_)
                scaler.update()
            else:
                total_loss.backward()
                for o_ in optimizer.values():
                    o_.step()
        else:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale_a.clamp_(0, math.log(100))
            if args.clap_mlploss:
                unwrap_model(model).logit_scale_t.clamp_(0, math.log(100))
                unwrap_model(model).logit_scale_v.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(audios)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Note: loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar_a = logit_scale_a.item()
            logit_scale_scalar_t = logit_scale_t.item()
            logit_scale_scalar_v = logit_scale_v.item()
            if isinstance(optimizer, dict):
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} "
                    f"Logit Scale Audio: {logit_scale_scalar_a:.3f} "
                    f"Logit Scale Text: {logit_scale_scalar_t:.3f} "
                    f"Logit Scale Vocal: {logit_scale_scalar_v:.3f}"
                )
                log_data = {
                    "loss": loss_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "scale_audio": logit_scale_scalar_a,
                    "scale_text": logit_scale_scalar_t,
                    "scale_vocal": logit_scale_scalar_v,
                    "lr": [o_.param_groups[0]["lr"] for o_ in optimizer.values()],
                }
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale Audio: {logit_scale_scalar_a:.3f} "
                    f"Logit Scale Text: {logit_scale_scalar_t:.3f} "
                    f"Logit Scale Vocal: {logit_scale_scalar_v:.3f}"
                )
                log_data = {
                    "loss": loss_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "scale_audio": logit_scale_scalar_a,
                    "scale_text": logit_scale_scalar_t,
                    "scale_vocal": logit_scale_scalar_v,
                    "lr": optimizer.param_groups[0]["lr"],
                }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name: val, "step": step})

            # Resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # End for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not args.parallel_eval:
        if not is_master(args):
            return metrics
    device = torch.device(args.device)
    model.eval()

    if is_master(args):
        print('Evaluating...')
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress()

    if "val" in data and (
            args.val_frequency
            and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        eval_info = {}
        eval_info["all"] = {
            "cumulative_loss": 0.0,
            "num_samples": 0,
            "all_audio_features": [],
            "all_text_features": [],
            "all_vocal_features": [],
            "all_audio_features_mlp": [],
            "all_text_features_mlp": [],
            "all_vocal_features_mlp": [],
        }

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audios = batch["waveform"]
                texts = batch["text"]
                vocals = batch["vocal_waveform"]

                with autocast():
                    (
                        audio_features,
                        text_features,
                        vocal_features,
                        audio_features_mlp,
                        text_features_mlp,
                        vocal_features_mlp,
                        logit_scale_a,
                        logit_scale_t,
                        logit_scale_v,
                    ) = model(audios, texts, vocals, device)

                    if args.parallel_eval:
                        (
                            audio_features,
                            text_features,
                            vocal_features,
                            audio_features_mlp,
                            text_features_mlp,
                            vocal_features_mlp,
                        ) = gather_features(
                            audio_features=audio_features,
                            text_features=text_features,
                            vocal_features=vocal_features,
                            audio_features_mlp=audio_features_mlp,
                            text_features_mlp=text_features_mlp,
                            vocal_features_mlp=vocal_features_mlp,
                            local_loss=False,
                            gather_with_grad=False,
                            rank=args.rank,
                            world_size=args.world_size,
                            use_horovod=args.horovod,
                            mlp_loss=args.clap_mlploss,
                        )

                    if is_master(args):
                        num_samples += audio_features.shape[0]
                        eval_info["all"]["all_audio_features"].append(
                            audio_features.cpu()
                        )
                        eval_info["all"]["all_text_features"].append(
                            text_features.cpu()
                        )
                        eval_info["all"]["all_vocal_features"].append(
                            vocal_features.cpu()
                        )
                        if args.clap_mlploss:
                            eval_info["all"]["all_audio_features_mlp"].append(
                                audio_features_mlp.cpu()
                            )
                            eval_info["all"]["all_text_features_mlp"].append(
                                text_features_mlp.cpu()
                            )
                            eval_info["all"]["all_vocal_features_mlp"].append(
                                vocal_features_mlp.cpu()
                            )

                    if is_master(args) and (i % 100) == 0:
                        logging.info(
                            f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]"
                        )

            if is_master(args):
                if args.clap_mlploss:
                    metrics = get_metrics(
                        audio_features=torch.cat(
                            eval_info["all"]["all_audio_features"]
                        ),
                        text_features=torch.cat(
                            eval_info["all"]["all_text_features"]
                        ),
                        vocal_features=torch.cat(
                            eval_info["all"]["all_vocal_features"]
                        ),
                        logit_scale_a=logit_scale_a.cpu(),
                        logit_scale_t=logit_scale_t.cpu(),
                        logit_scale_v=logit_scale_v.cpu(),
                        audio_features_mlp=torch.cat(
                            eval_info["all"]["all_audio_features_mlp"]
                        ),
                        text_features_mlp=torch.cat(
                            eval_info["all"]["all_text_features_mlp"]
                        ),
                        vocal_features_mlp=torch.cat(
                            eval_info["all"]["all_vocal_features_mlp"]
                        ),
                        mlp_loss=args.clap_mlploss,
                    )
                else:
                    metrics = get_metrics(
                        audio_features=torch.cat(
                            eval_info["all"]["all_audio_features"]
                        ),
                        text_features=torch.cat(
                            eval_info["all"]["all_text_features"]
                        ),
                        vocal_features=torch.cat(
                            eval_info["all"]["all_vocal_features"]
                        ),
                        logit_scale_a=logit_scale_a.cpu(),
                        logit_scale_v=logit_scale_v.cpu(),
                        mlp_loss=args.clap_mlploss,
                    )
                metrics.update({"epoch": epoch})

    if is_master(args):
        if not metrics:
            return metrics

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\n".join(
                [f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]
            )
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, "Please install wandb."
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, "epoch": epoch})

        return metrics
    else:
        return metrics


def get_metrics(
        audio_features,
        text_features,
        vocal_features,
        logit_scale_a,
        logit_scale_v,
        logit_scale_t=None,
        audio_features_mlp=None,
        text_features_mlp=None,
        vocal_features_mlp=None,
        mlp_loss=False,
):
    metrics = {}
    if mlp_loss:
        # Compute logits for each modality pair
        a_t_logits = (
            (logit_scale_a * audio_features @ text_features_mlp.t()).detach().cpu()
        )
        t_a_logits = a_t_logits.t()
        a_v_logits = (
            (logit_scale_v * audio_features @ vocal_features_mlp.t()).detach().cpu()
        )
        v_a_logits = a_v_logits.t()
        t_v_logits = (
            (logit_scale_t * text_features @ vocal_features_mlp.t()).detach().cpu()
        )
        v_t_logits = t_v_logits.t()

        labels = torch.arange(audio_features.shape[0]).long()
        # Compute cross-entropy loss for each modality pair
        total_loss = (
                             F.cross_entropy(a_t_logits, labels)
                             + F.cross_entropy(t_a_logits, labels)
                             + F.cross_entropy(a_v_logits, labels)
                             + F.cross_entropy(v_a_logits, labels)
                             + F.cross_entropy(t_v_logits, labels)
                             + F.cross_entropy(v_t_logits, labels)
                     ) / 6

        metrics["cumulative_loss"] = total_loss.item()
        metrics["num_samples"] = audio_features.shape[0]

        logits = {
            "audio_to_text": a_t_logits,
            "text_to_audio": t_a_logits,
            "audio_to_vocal": a_v_logits,
            "vocal_to_audio": v_a_logits,
            "text_to_vocal": t_v_logits,
            "vocal_to_text": v_t_logits,
        }
    else:
        # Compute logits for each modality pair
        a_t_logits = (logit_scale_a * audio_features @ text_features.t()).detach().cpu()
        t_a_logits = a_t_logits.t()
        a_v_logits = (logit_scale_v * audio_features @ vocal_features.t()).detach().cpu()
        v_a_logits = a_v_logits.t()

        labels = torch.arange(audio_features.shape[0]).long()
        total_loss = (
                             F.cross_entropy(a_t_logits, labels)
                             + F.cross_entropy(t_a_logits, labels)
                             + F.cross_entropy(a_v_logits, labels)
                             + F.cross_entropy(v_a_logits, labels)
                     ) / 4

        metrics["cumulative_loss"] = total_loss.item()
        metrics["num_samples"] = audio_features.shape[0]

        logits = {
            "audio_to_text": a_t_logits,
            "text_to_audio": t_a_logits,
            "audio_to_vocal": a_v_logits,
            "vocal_to_audio": v_a_logits,
        }

    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        # mAP@10
        metrics[f"{name}_mAP@10"] = np.mean(
            np.where(preds < 10, 1 / (preds + 1), 0.0)
        )

    return metrics
