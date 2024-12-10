import argparse


def get_default_params(model_name):
    # Params from CLIP paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to h5 file with training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to h5 file with validation data",
    )
    parser.add_argument(
        "--train-ipc",
        type=str,
        default=None,
        help="Path to npy file of the number of instances per class in training data",
    )
    parser.add_argument(
        "--val-ipc",
        type=str,
        default=None,
        help="Path to npy file of the number of instances per class in validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in training dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in validation dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "auto", "toy"],
        default="auto",
        help="Which type of dataset to process.",
    )
    parser.add_argument(
        "--datasetnames",
        nargs="+",
        default=None,
        help="If loading webdataset, specify the dataset names to load.",
    )
    parser.add_argument(
        "--full-train-dataset",
        nargs="+",
        default=None,
        help="Which dataset will be trained with all the subsets (train+test).",
    )
    parser.add_argument(
        "--exclude-eval-dataset",
        nargs="+",
        default=None,
        help="Which dataset will be excluded from evaluation.",
    )
    parser.add_argument(
        "--datasetinfos",
        nargs="+",
        default=None,
        help="If loading webdataset, specify the dataset types to load.",
    )
    parser.add_argument(
        "--dataset-proportion",
        type=float,
        default=1.0,
        help="Proportion of the dataset to use for training.",
    )
    parser.add_argument(
        "--remotedata",
        default=False,
        action="store_true",
        help="If the dataset is remote, set this flag.",
    )
    parser.add_argument(
        "--class-label-path",
        type=str,
        default=None,
        help="The path of the class label pickle or csv.",
    )
    parser.add_argument(
        "--datasetpath",
        type=str,
        default="/mnt/audio_clip/webdataset_tar",
        help="The path to the dataset.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="Log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")

    # Split optimizer parameters for pretrained and new parameters
    parser.add_argument(
        "--split-opt",
        action="store_true",
        default=False,
        help="Use this flag to use different optimizer settings for pretrained and new parameters.",
    )
    parser.add_argument(
        "--lr-pretrained", type=float, default=None, help="Learning rate for pretrained parameters."
    )
    parser.add_argument(
        "--beta1-pretrained", type=float, default=None, help="Adam beta 1 for pretrained parameters."
    )
    parser.add_argument(
        "--beta2-pretrained", type=float, default=None, help="Adam beta 2 for pretrained parameters."
    )
    parser.add_argument(
        "--eps-pretrained", type=float, default=None, help="Adam epsilon for pretrained parameters."
    )
    parser.add_argument(
        "--wd-pretrained", type=float, default=0.2, help="Weight decay for pretrained parameters."
    )
    parser.add_argument(
        "--momentum-pretrained", type=float, default=0.9, help="Momentum for pretrained parameters."
    )
    parser.add_argument(
        "--lr-new", type=float, default=None, help="Learning rate for new parameters."
    )
    parser.add_argument(
        "--beta1-new", type=float, default=None, help="Adam beta 1 for new parameters."
    )
    parser.add_argument(
        "--beta2-new", type=float, default=None, help="Adam beta 2 for new parameters."
    )
    parser.add_argument(
        "--eps-new", type=float, default=None, help="Adam epsilon for new parameters."
    )
    parser.add_argument(
        "--wd-new", type=float, default=0.2, help="Weight decay for new parameters."
    )
    parser.add_argument(
        "--momentum-new", type=float, default=0.9, help="Momentum for new parameters."
    )
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-top-performance",
        type=int,
        default=0,
        help="Save the top x performance weights if the value > 0.",
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val data.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to latest checkpoint (default: none).",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # Model parameters
    parser.add_argument(
        "--amodel",
        type=str,
        default="RN50",
        help="Name of the audio backbone to use.",
    )
    parser.add_argument(
        "--tmodel",
        type=str,
        default="transformer",
        help="Name of the text backbone to use. Can be [transformer, bert, roberta, bart].",
    )
    parser.add_argument(
        "--vmodel",
        type=str,
        default="RN50",
        help="Name of the vocal backbone to use.",
    )
    parser.add_argument(
        "--pretrained-audio",
        default="",
        type=str,
        help="Use pretrained weights for the audio encoder of CLVAP.",
    )
    parser.add_argument(
        "--pretrained-text",
        default="",
        type=str,
        help="Use pretrained weights for the text encoder of CLVAP.",
    )
    parser.add_argument(
        "--pretrained-vocal",
        default="",
        type=str,
        help="Use pretrained weights for the vocal encoder of CLVAP.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="Use pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--freeze-text",
        default=False,
        action="store_true",
        help="If you need to freeze the text encoder, set this flag.",
    )
    parser.add_argument(
        "--freeze-text-after",
        type=int,
        default=-1,
        help="If you need to freeze the text encoder after (include) epoch x, set this param to x. Set -1 to disable it.",
    )
    parser.add_argument(
        "--freeze-vocal",
        default=False,
        action="store_true",
        help="If you need to freeze the vocal encoder, set this flag.",
    )
    parser.add_argument(
        "--freeze-vocal-after",
        type=int,
        default=-1,
        help="If you need to freeze the vocal encoder after (include) epoch x, set this param to x. Set -1 to disable it.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action="store_true",
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action="store_true",
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="Calculate loss with local features @ global (instead of realizing full global @ global matrix).",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="Enable full distributed gradient for feature gather.",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action="store_true",
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="Torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="Torch.jit.trace the model for inference/eval only.",
    )
    # Distributed training parameters
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="URL used to set up distributed training.",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="Distributed backend."
    )
    parser.add_argument(
        "--report-to",
        default="",
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard'].",
    )
    parser.add_argument(
        "--wandb-notes", default="", type=str, help="Notes if logging with wandb."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, copy the entire codebase to the log directory and execute from there.",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use Horovod for distributed training.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--seed", type=int, default=4242, help="Default random seed.")

    # Checkpoint selection parameters
    parser.add_argument(
        "--top-k-checkpoint-select-dataset",
        type=str,
        default="all",
        help="The dataset for selecting top-k checkpoint.",
    )
    parser.add_argument(
        "--top-k-checkpoint-select-metric",
        type=str,
        default="_R@10",
        help="The metric for selecting top-k checkpoint.",
    )
    parser.add_argument(
        "--openai-model-cache-dir",
        type=str,
        default="~/.cache/clip",
        help="Directory to download OpenAI models.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer type, can be 'adamw' or 'sgd'.",
    )
    parser.add_argument(
        "--parallel-eval",
        default=False,
        action="store_true",
        help="Evaluate in parallel (multi-GPU, multi-node).",
    )
    parser.add_argument(
        "--no-eval",
        default=False,
        action="store_true",
        help="Training without evaluation.",
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="The ID of wandb experiment to restore.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0, help="Sleep n seconds before start training."
    )
    # Variable length processing
    parser.add_argument(
        "--enable-fusion",
        default=False,
        action="store_true",
        help="Enable feature fusion for variable-length data.",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default='None',
        help="Fusion type among ['channel_map', 'daf_1d', 'aff_1d', 'iaff_1d', 'daf_2d', 'aff_2d', 'iaff_2d'].",
    )
    parser.add_argument(
        "--mixup",
        default=False,
        action="store_true",
        help="Enable mixup in finetuning training.",
    )
    parser.add_argument(
        "--text-augment-selection",
        type=str,
        default=None,
        help="For selecting levels of augmented text. Options are ['all', 'augment_only', 'none'].",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="The prefetch factor for dataloader. Larger value will use more memory and CPU but faster.",
    )
    parser.add_argument(
        "--data-filling",
        type=str,
        default="pad",
        help="Type of data filling when the audio length is shorter than the max length. Options are ['repeat', 'repeatpad', 'pad'].",
    )
    parser.add_argument(
        "--data-truncating",
        type=str,
        default="rand_trunc",
        help="Type of data truncation when the audio length is longer than the max length. Options are ['rand_trunc', 'fusion'].",
    )
    # Linear Probe parameters
    parser.add_argument(
        "--lp-mlp",
        default=False,
        action="store_true",
        help="Use MLP layer for Linear Probe.",
    )
    parser.add_argument(
        "--lp-freeze",
        default=False,
        action="store_true",
        help="Freeze CLVAP model during Linear Probe.",
    )
    parser.add_argument(
        "--lp-act",
        default="None",
        type=str,
        help="Activation function for Linear Probe. Options are ['relu', 'elu', 'prelu', 'softmax', 'sigmoid'].",
    )
    parser.add_argument(
        "--lp-loss", type=str, default="bce", help="Loss function for Linear Probe."
    )
    parser.add_argument(
        "--lp-metrics",
        type=str,
        default="map,mauc,acc",
        help="Metrics for Linear Probe.",
    )
    parser.add_argument(
        "--lp-lr", type=float, default=1e-4, help="Learning rate for Linear Probe."
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0,
        help="The kappa in the weighted contrastive loss. Default is 0 (turn off weighted contrastive loss).",
    )
    parser.add_argument(
        "--clap-mlploss",
        default=False,
        action="store_true",
        help="Use MLP loss for CLVAP model.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="The prefetch factor for dataloader. Larger value will use more memory and CPU but faster.",
    )

    args = parser.parse_args()

    # If some params are not passed, use the default values based on model name.
    default_params = get_default_params(args.amodel)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
