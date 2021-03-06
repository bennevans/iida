
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def make_encoder_oh(input_size, N=1, d_model=320, h=5, dropout=0.1, \
                    d_emb=8, use_extra_fc=True, no_norm=False, device='cpu'):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model).to(device)
    model = EmbedEncoder(
                Encoder(EncoderLayer(d_model, c(attn), dropout,  no_norm=no_norm).to(device), \
                    d_model=d_model, d_emb=d_emb, use_extra_fc=use_extra_fc, no_norm=no_norm
                ).to(device),
                LinearEmb(d_model, input_size).to(device),
                device=device
            ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class EmbedEncoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, src_embed, device='cpu'):
        super(EmbedEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.device = device

    def forward(self, datum, src_mask=None):
        s = datum['context_obs'].to(self.device)
        a = datum['context_act'].to(self.device)
        sp = datum['context_obs_prime'].to(self.device)
        x = torch.cat([s, a, sp], dim=-1).float()
        return self.encoder(self.src_embed(x), src_mask)


class LinearEmb(nn.Module):
    def __init__(self, d_model, input_size):
        super(LinearEmb, self).__init__()
        self.lin_emb = nn.Linear(input_size, d_model)

    def forward(self, x):
        return self.lin_emb(x.float())


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, d_model=320, d_emb=32, use_extra_fc=False, no_norm=False):
        super(Encoder, self).__init__()
        self.layer = layer
        self.norm = LayerNorm(layer.size)
        self.use_extra_fc = use_extra_fc
        self.no_norm = no_norm
        if self.use_extra_fc:
            self.fc_out = nn.Linear(d_model, d_emb)

    def forward(self, x, mask=None, use_extra_fc=False):
        "Pass the input (and mask) through each layer in turn."
        x = self.layer(x, mask)
        if not self.no_norm:
            x = self.norm(x)
        if self.use_extra_fc:
            x = self.fc_out(x.squeeze(1).mean(1))
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, dropout, no_norm=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size
        self.no_norm = no_norm

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        if self.no_norm:
            x = self.self_attn(x, x, x, mask)
        else:
            x = self.sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        return  self.linears[-1](x)
