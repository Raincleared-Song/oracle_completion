import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """
    token embedding
    referenced to https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/token.py
    """
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
