import torch
import torch.nn as nn
from transformers import PreTrainedModel

from basic_boltzgen import Boltz
from boltzgen_config import BoltzGenConfig


class BoltzGen(PreTrainedModel):
    config_class = BoltzGenConfig
    def __init__(self, config: BoltzGenConfig):
        super().__init__(config)
        self.config = config

        self.boltz = Boltz(**config.__dict__)