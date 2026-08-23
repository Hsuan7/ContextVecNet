"""
This script sets up and runs a custom CLIP model evaluation pipeline.
It includes functions for configuration loading, model loading, and custom data collation.
Additionally, it defines a Config class for dot-notation access to dictionaries,
fixes the random seed for reproducibility, and processes command-line arguments.
"""

import warnings
import argparse
import torch

# Set the multiprocessing sharing strategy for torch
torch.multiprocessing.set_sharing_strategy("file_descriptor")
import wandb
import pprint
import yaml
import os
from clip import clip
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from maple_model import CustomCLIP, TextOnlyBERT

from utils import load_args, load_checkpoint

from particular_model_trainers import Trainer

import callbacks
from trainer import NotALightningTrainer
from loggers import WandbLogger

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import nomenclature

from datasets.twitter_learn import CombinedTwitterDataset

# Set the device to "cuda" if available, otherwise "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


def fix_seed(seed):
    """
    Fix the seed for reproducibility across random, numpy, and torch.

    This function sets the seed for Python's random module, NumPy, and PyTorch.
    It also ensures deterministic behavior for CUDA operations.

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
    A dictionary subclass that allows attribute-style access.

    This class recursively converts nested dictionaries to Config objects,
    enabling dot notation to access dictionary keys.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Config object and recursively convert nested dicts.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(Config, self).__init__(*args, **kwargs)
        # Recursively convert dicts to Config objects
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, attr):
        """
        Enable attribute-style access for dictionary keys.

        Args:
            attr (str): The attribute to access.

        Returns:
            The value corresponding to the given attribute key.
        """
        return self.get(attr)

    def __setattr__(self, key, value):
        """
        Enable setting dictionary values using attribute-style access.

        Args:
            key (str): The attribute name.
            value: The value to set.
        """
        self[key] = value

    def __delattr__(self, key):
        """
        Enable deletion of dictionary keys using attribute-style access.

        Args:
            key (str): The attribute name to delete.
        """
        del self[key]


def load_config(yaml_path):
    """
    Load a YAML configuration file and return a Config object for dot notation access.

    Args:
        yaml_path (str): The file path to the YAML configuration file.

    Returns:
        Config: A Config object containing the configuration parameters.
    """
    with open(yaml_path, "r") as file:
        cfg_dict = yaml.safe_load(file)
    return Config(cfg_dict)


def load_clip_to_cpu(cfg):
    """
    Load the CLIP model onto CPU and build a custom CLIP model with design details.

    The function retrieves the backbone name from the configuration, downloads the model,
    and attempts to load a JIT archive. If that fails, it loads the state dictionary.
    It then builds the model with specified design details from the configuration.

    Args:
        cfg (Config): Configuration object with model parameters.

    Returns:
        model: The loaded and built CLIP model.
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
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
    Custom collate function for handling variable size images in the dataset.

    This function processes a batch of samples by extracting and converting various fields
    into tensors where appropriate. Images are kept as a list to allow for variable sizes.

    Args:
        batch (list): A list of sample dictionaries from the dataset.

    Returns:
        dict: A dictionary containing collated data with the following keys:
            - "author": List of user names.
            - "images": List of images (kept as a list).
            - "images_paths": List of image paths.
            - "texts": List of text entries.
            - "time": Tensor of timestamps.
            - "label": Tensor of labels reshaped to (-1, 1) on the specified device.
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

    # Convert labels and masks to tensors
    labels_tensor = torch.tensor(labels, dtype=torch.float).clone().detach()
    text_mask_tensor = torch.tensor(text_mask, dtype=torch.float).clone().detach()
    image_mask_tensor = torch.tensor(image_mask, dtype=torch.float).clone().detach()
    dates_tensor = torch.tensor(dates, dtype=torch.float).clone().detach()

    # Create a sample dictionary
    sample = {
        "author": user_names,
        "images": images,  # Keep images as a list of variable sizes
        "images_paths": images_paths,
        "texts": texts,
        "time": dates_tensor,
        "label": labels_tensor.to(device).view(-1, 1),
        "padding_amount": padding_amount,
        "text_mask": text_mask_tensor,
        "image_mask": image_mask_tensor,
    }
    # # ===== modality ablation =====
    if args.modality == "text":
        sample["image_mask"] = torch.zeros_like(sample["image_mask"])

    elif args.modality == "image":
        sample["text_mask"] = torch.zeros_like(sample["text_mask"])
    return sample


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Custom CLIP evaluation pipeline.")
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--group", type=str, default="default")
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--mode", type=str, default="dryrun")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="reddit")
parser.add_argument("--log_every", type=int, default=5)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--fold", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--position_embeddings", type=str, default=None)
parser.add_argument("--image_embeddings_type", type=str, default=None)
parser.add_argument("--text_embeddings_type", type=str, default=None)
parser.add_argument("--config_file", type=str, default="configs/combos/clip_clip.yaml")

args = parser.parse_args()
args, cfg = load_args(args)

# Load main configuration file
cfg1 = load_config("configs/maple.yaml")

if args.text_embeddings_type == "bert":
    print(f"Building text-only BERT: {args.bert_model_name}")
    model = TextOnlyBERT(args)
else:
    clip_model = load_clip_to_cpu(cfg1)
    print(
        "---------------------------------------------------------------------------------------------------------"
    )
    if cfg1.TRAINER.MAPLE.PREC == "fp32" or cfg1.TRAINER.MAPLE.PREC == "amp":
        clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(cfg1, clip_model, args)

# Retrieve the dataset based on the provided arguments
dataset = nomenclature.DATASETS[args.dataset]

# Load the model state dictionary and validation-time calibration settings.
checkpoint = load_checkpoint(args)
checkpoint_metrics = checkpoint.get("metrics", {})
args.eval_calibration = {
    "platt_a": checkpoint_metrics.get("val/platt_a"),
    "platt_b": checkpoint_metrics.get("val/platt_b"),
    "threshold": checkpoint_metrics.get("val/best_threshold"),
}
print("::: Evaluation calibration from validation:", args.eval_calibration)

state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

# Load state dict into the model, set evaluation mode, and move to GPU if available
model.load_state_dict(state_dict)
model.eval()
model.train(False)
model.cuda()
model.float()

# Instantiate the evaluator from nomenclature and evaluate the model
evaluator = nomenclature.EVALUATORS["multimodal-evaluator"](args, model)
results = evaluator.evaluate(save=True)
print(evaluator.__class__.__name__)
pprint.pprint(results)
