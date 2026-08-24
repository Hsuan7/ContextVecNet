import yaml
from datasets import *
from models import *
from evaluators import MultimodalEvaluator

import torch

device = torch.device("cuda")

DATASETS = {
    "twitter": CombinedTwitterDataset,
}

EVALUATORS = {
    "multimodal-evaluator": MultimodalEvaluator,
}

MODELS = {
    "multimodal-transformer": MultiModalTransformer,
    "text-image-concat": TextImageConcatBaseline,
    "lstm-baseline": LSTMBaseline,
}
