"""
This script sets up and runs a training pipeline for a custom CLIP model on a multimodal
Twitter dataset. It includes functions for configuration loading, model loading, and
data collation, as well as setting up training utilities such as callbacks, loggers, and
learning rate schedulers.
"""

import warnings
import argparse
import torch

# Set the multiprocessing sharing strategy for torch to use file descriptors
torch.multiprocessing.set_sharing_strategy("file_descriptor")
import wandb
import pprint
import yaml
import os
from clip import clip
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from maple_model import BertImageCLIP, CustomCLIP, TextOnlyBERT

from utils import load_args

from particular_model_trainers import Trainer

import callbacks
from trainer import NotALightningTrainer
from loggers import WandbLogger

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import nomenclature

from datasets.twitter_learn import CombinedTwitterDataset
from datasets.window_level_dataset import WindowLevelDataset
# Set the device to "cuda" if available, otherwise "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


def fix_seed(seed):
    """
    Fix the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function sets the seed for Python's random module, NumPy, and PyTorch, and
    ensures deterministic behavior for CUDA operations.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Fix the seed for reproducibility
fix_seed(42)


class Config(dict):
    """
    A dictionary subclass that provides attribute-style access to its keys.

    This class recursively converts nested dictionaries into Config objects, allowing
    dot notation to access dictionary items.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Config object and recursively convert nested dictionaries.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(Config, self).__init__(*args, **kwargs)
        # Recursively convert dicts to Config objects for dot notation access
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, attr):
        """
        Enable attribute-style access to dictionary items.

        Args:
            attr (str): The attribute name to access.

        Returns:
            The value corresponding to the attribute if it exists, else None.
        """
        return self.get(attr)

    def __setattr__(self, key, value):
        """
        Enable attribute-style setting of dictionary items.

        Args:
            key (str): The attribute name.
            value: The value to set.
        """
        self[key] = value

    def __delattr__(self, key):
        """
        Enable attribute-style deletion of dictionary items.

        Args:
            key (str): The attribute name to delete.
        """
        del self[key]


def load_config(yaml_path):
    """
    Load a YAML configuration file and return a Config object with dot notation access.

    Args:
        yaml_path (str): The file path to the YAML configuration file.

    Returns:
        Config: A configuration object containing the loaded parameters.
    """
    with open(yaml_path, "r") as file:
        cfg_dict = yaml.safe_load(file)
    return Config(cfg_dict)


def load_clip_to_cpu(cfg):
    """
    Load the CLIP model onto the CPU and build a custom CLIP model with design details.

    The function retrieves the backbone name from the configuration, downloads the model,
    and attempts to load it as a JIT archive. If that fails, it loads the state dictionary.
    It then builds the model with the provided design details.

    Args:
        cfg (Config): Configuration object containing model parameters.

    Returns:
        model: The loaded and built CLIP model.
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # Try loading the JIT archive of the model
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # If JIT loading fails, load the state dictionary
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE.N_CTX,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def custom_collate_fn(batch):
    """
    Custom collate function for handling variable-size images in a dataset.

    This function processes a batch of samples by extracting data fields, converting
    appropriate fields into tensors, and returning a dictionary with collated data.
    Images are retained as a list to support variable sizes.

    Args:
        batch (list): A list of sample dictionaries from the dataset.

    Returns:
        dict: A dictionary containing the collated batch with keys:
            - "author": List of user names.
            - "images": List of images.
            - "images_paths": List of image paths.
            - "texts": List of text entries.
            - "time": Tensor of timestamps.
            - "label": Tensor of labels reshaped to (-1, 1) on the designated device.
            - "padding_amount": List of padding amounts.
            - "text_mask": Tensor of text masks.
            - "image_mask": Tensor of image masks.
    """
    # Extract items from the batch
    user_names = [item["author"] for item in batch]
    labels = [item["label"] for item in batch]
    texts = [item["texts"] for item in batch]
    dates = [item["time"] for item in batch]
    images = [item["images"] for item in batch]
    images_paths = [item["images_paths"] for item in batch]
    padding_amount = [item["padding_amount"] for item in batch]
    text_mask = [item["text_mask"] for item in batch]
    image_mask = [item["image_mask"] for item in batch]

    # Convert lists to tensors
    labels_tensor = torch.tensor(labels, dtype=torch.float).clone().detach()
    text_mask_tensor = torch.tensor(
        text_mask, dtype=torch.float).clone().detach()
    image_mask_tensor = torch.tensor(
        image_mask, dtype=torch.float).clone().detach()
    dates_tensor = torch.tensor(dates, dtype=torch.float).clone().detach()

    # Create a sample dictionary
    sample = {
        "author": user_names,
        "images": images,  # Images remain as a list of variable sizes
        "images_paths": images_paths,
        "texts": texts,
        "time": dates_tensor,
        "label": labels_tensor.to(device).view(-1, 1),
        "padding_amount": padding_amount,
        "text_mask": text_mask_tensor,
        "image_mask": image_mask_tensor,
    }
    # ===== modality ablation =====
    if args.modality == "text":
        sample["image_mask"] = torch.zeros_like(sample["image_mask"])

    elif args.modality == "image":
        sample["text_mask"] = torch.zeros_like(sample["text_mask"])



    return sample


# 改
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Custom CLIP training pipeline.")
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--group", type=str, default="default")
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--mode", type=str, default="dryrun")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1)  # 256
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--log_every", type=int, default=5)
parser.add_argument("--early_stopping_patience", type=int, default=50)
parser.add_argument("--use_class_weights", action="store_true", default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--fold", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--position_embeddings", type=str, default=None)
parser.add_argument("--image_embeddings_type", type=str, default=None)
parser.add_argument("--text_embeddings_type", type=str, default=None)
parser.add_argument(
    "--config_file", type=str, default="configs/combos/clip_clip.yaml"
)
parser.add_argument(
    "--resume", type=str, default=None, help="Path to resume checkpoint"
)
# 改 成window_level下面兩行
# parser.add_argument("--window_dataset_csv", type=str,
#                     default="/mnt/d/時間序列/window_level_plus_prev_context_8/window_dataset.csv")
# parser.add_argument("--post_level_csv", type=str,
#                     default="/mnt/d/時間序列/labeled/post_with_user_label.csv")
args = parser.parse_args()
args, cfg = load_args(args)
print("===== MODALITY CHECK =====")
print("config_file:", args.config_file)
print("args.modality:", args.modality)
print("==========================")
# Load configuration from YAML file
cfg1 = load_config("configs/maple.yaml")

if args.text_embeddings_type == "bert" and args.modality == "text":
    print(f"Building text-only BERT: {args.bert_model_name}")
    model = TextOnlyBERT(args)
elif args.text_embeddings_type == "bert" and args.modality == "both":
    clip_model = load_clip_to_cpu(cfg1)
    if cfg1.TRAINER.MAPLE.PREC == "fp32" or cfg1.TRAINER.MAPLE.PREC == "amp":
        clip_model.float()
    print(f"Building BERT+Image CLIP ContextVecNet: {args.bert_model_name}")
    model = BertImageCLIP(cfg1, clip_model, args)
else:
    # Preserve the original ContextVecNet-CLIP/CLIP construction path.
    clip_model = load_clip_to_cpu(cfg1)
    print(
        "---------------------------------------------------------------------------------------------------------"
    )
    if cfg1.TRAINER.MAPLE.PREC == "fp32" or cfg1.TRAINER.MAPLE.PREC == "amp":
        clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(cfg1, clip_model, args)

# Set model to float precision
model.float()

print("Turning off gradients in both the image and the text encoder")

# Define parameter names that should remain trainable
name_to_update = [
    "prompt_learner.compound_prompt_projections",
    "prompt_learner.compound_prompts_text",
    "prompt_learner.proj",
    "prompt_learner.ctx",
    "multi_modal_transformer",
]
if args.text_embeddings_type == "bert" and args.bert_finetune:
    name_to_update.append("bert_text_encoder")

# Adjust parameter gradients: disable for parameters not matching the update list,
# except for those containing "VPT" which are always trainable.
for name, param in model.named_parameters():
    if not any(substring in name for substring in name_to_update):
        if "VPT" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    else:
        # Ensure parameters in the update list are trainable
        param.requires_grad_(True)

# Print names of parameters that will be updated
enabled = set()
for name, param in model.named_parameters():
    if param.requires_grad:
        enabled.add(name)
print(f"Parameters to be updated: {enabled}")

# Print parsed arguments for verification
pprint.pprint(args.__dict__)

# Set environment variables for Weights & Biases (wandb) logging
os.environ["WANDB_MODE"] = args.mode
os.environ["WANDB_NAME"] = args.name
os.environ["WANDB_NOTES"] = args.notes

# Initialize wandb logging
wandb.init(project="multimodal-depression-time",
           group=args.group, entity="blue-erisk")
wandb.config.update(vars(args))
wandb.config.update({"config": cfg})

NUM_WORKERS = 2

# Retrieve dataset from nomenclature based on command-line argument
dataset = nomenclature.DATASETS[args.dataset]

# Create training and validation datasets
# 改 成window_level下面兩行
train_dataset = CombinedTwitterDataset(args=args, kind="train")
val_dataset = CombinedTwitterDataset(args=args, kind="valid")
# train_dataset = WindowLevelDataset(args=args, kind="train")
# val_dataset = WindowLevelDataset(args=args, kind="valid")

# Compute positive-class weight for BCEWithLogitsLoss if required.
labels = np.array(train_dataset.labels, dtype=np.float32)
num_positive = float(labels.sum())
num_negative = float(len(labels) - num_positive)
pos_weight = num_negative / num_positive if num_positive > 0 else None
print("num_positive =", int(num_positive))
print("num_negative =", int(num_negative))
print("pos_weight =", pos_weight)
# 改
# labels = np.array(train_dataset.labels)

# print("len(train_dataset) =", len(train_dataset))
# print("len(val_dataset) =", len(val_dataset))
# print("labels sample =", labels[:10])
# print("labels dtype before =", labels.dtype)
# print("unique before =", np.unique(labels))

# label_map = {
#     "negative": 0,
#     "positive": 1,
#     "0": 0,
#     "1": 1,
#     0: 0,
#     1: 1,
#     0.0: 0,
#     1.0: 1,
# }

# clean_labels = []
# for x in labels:
#     if x in label_map:
#         clean_labels.append(label_map[x])
#     else:
#         raise ValueError(f"Unexpected label value: {x} (type={type(x)})")

# labels = np.array(clean_labels, dtype=np.int64)
# classes = np.unique(labels).astype(np.int64)

# print("labels dtype after =", labels.dtype)
# print("unique after =", classes)

# class_weights = compute_class_weight(
#     class_weight="balanced",
#     classes=classes,
#     y=labels,
# )

# Initialize the trainer with model, arguments, and class weights (if used)
trainer = Trainer(
    args, model, pos_weight=pos_weight if args.use_class_weights else None
)

# Create DataLoaders for training and validation datasets with the custom collate function
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn
)

# Initialize wandb logger for experiment tracking
wandb_logger = WandbLogger()

batch = next(iter(train_dataloader))

print("===== BATCH CHECK =====")
print("authors:", batch["author"][:2])
print("label shape:", batch["label"].shape)
print("time shape:", batch["time"].shape)
print("text_mask shape:", batch["text_mask"].shape)
print("image_mask shape:", batch["image_mask"].shape)

print("num batch images:", len(batch["images"]))
print("num images per user:", len(batch["images"][0]))
print("image tensor shape:", batch["images"][0][0].shape)

print("num batch texts:", len(batch["texts"]))
print("num texts per user:", len(batch["texts"][0]))
print("first text:", batch["texts"][0][0])

print("first 5 image paths:", batch["images_paths"][0][:5])
print("first 5 time:", batch["time"][0][:5])
print("first 5 text_mask:", batch["text_mask"][0][:5])
print("first 5 image_mask:", batch["image_mask"][0][:5])




# Set up a model checkpoint callback to save checkpoints based on validation loss.
checkpoint_callback = callbacks.ModelCheckpoint(
    monitor="val_loss",
    direction="down",
    dirpath=f"checkpoints/{args.group}:{args.name}",
    save_weights_only=False,
    filename="epoch={epoch}-val_loss={val_loss:.6f}.ckpt",
)

# Configure a cyclic learning rate scheduler for the optimizer
# lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
#     optimizer=trainer.configure_optimizers(lr=args.base_lr),
#     cycle_momentum=False,
#     base_lr=args.base_lr,
#     mode="triangular",
#     step_size_up=10 * len(train_dataloader) /
#     args.accumulation_steps,  # per epoch
#     max_lr=args.base_lr * 10,
# )

# # Create a callback to update the learning rate at the end of each batch
# lr_callback = callbacks.LambdaCallback(
#     on_batch_end=lambda: lr_scheduler.step())

# # Create a callback to log the current learning rate to wandb at the end of each batch
# lr_logger = callbacks.LambdaCallback(
#     on_batch_end=lambda: wandb_logger.log("lr", lr_scheduler.get_last_lr()[0])
# )

lr_callback = callbacks.LambdaCallback(
    on_batch_end=lambda: None
)

lr_logger = callbacks.LambdaCallback(
    on_batch_end=lambda: wandb_logger.log("lr", args.base_lr)
)

early_stopping_callback = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=args.early_stopping_patience,
    direction="down",
)


# Initialize a custom trainer (NotALightningTrainer) with callbacks and logger
acumen_trainer = NotALightningTrainer(
    args=args,
    callbacks=[checkpoint_callback, early_stopping_callback, lr_callback, lr_logger],
    logger=wandb_logger,
)

# Resume training from a checkpoint if the resume argument is provided
if args.resume:
    checkpoint_path = args.resume
    checkpoint_callback.load_checkpoint(checkpoint_path)
    print(f"Resuming training from checkpoint at epoch {acumen_trainer.epoch}")

# Keep CUDA kernels reproducible across folds and reruns.
torch.backends.cudnn.benchmark = False

# Start the training process using the custom trainer
acumen_trainer.fit(trainer, train_dataloader, val_dataloader)
