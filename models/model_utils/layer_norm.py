import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    referenced to https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/layer_norm.py
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # weight
        self.b_2 = nn.Parameter(torch.zeros(features))  # bias
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
