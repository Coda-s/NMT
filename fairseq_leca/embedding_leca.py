import math
import torch
from torch import nn
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules import (
    MultiheadAttention,
    LearnedPositionalEmbedding,
)

def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m

class ConsPostionEmbedding(nn.Module):
    """
    positional embeddings for constraints.
    """
    def __init__(self, embedding_dim, padding_idx, init_size=20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = ConsPostionEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.positions = torch.tensor([])
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_cons, embedding_dim, padding_idx=None, startpos=1025):
        """Build sinusoidal embeddings for constraints.
        input: num_cons, number of constraints. 
               embedding_dim, dimension of embeddings
               startpos: start position of constraints, to differentiate the position of constraint from the normal src words.
        """   
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(startpos, startpos+num_cons, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_cons, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_cons, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, startpos=1025):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = ConsPostionEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
                startpos=startpos,
            )
        self.weights = self.weights.type_as(self._float_tensor)
        positions = self.get_positions(input, self.padding_idx)    
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def get_positions(self, tensor, padding_idx, startpos=0, tgt_tensor=None, sep_id=4):
        """Replace non-padding symbols with their position numbers.
        Position numbers begin at padding_idx+1. use constraint-right-pad-source=True
        padding_idx position=1, sep_idx position = 2 , others begin with 3, 
        a little different from the figure 2 in paper.
        """
        sep_cons = torch.ones_like(tensor)  
        bsz,clen = tensor.size()       
        for b in range(bsz):
            for j in range(clen):
                if tensor[b,j] == padding_idx:
                    break 
                sep_cons[b,j]= 2 if tensor[b,j] == sep_id else sep_cons[b,j-1]+1 
        return sep_cons
    
    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024, consnmt=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.consnmt=consnmt
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
        if seq_len == 0:
            return 
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = (timestep.int() + 1).long() if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights[self.padding_idx + pos, :].unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        positions = make_positions(input, self.padding_idx, self.left_pad, self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def make_positions(tensor, padding_idx, left_pad, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    if onnx_trace:
        range_buf = torch._dim_arange(like=tensor, dim=1) + padding_idx + 1
        mask = tensor.ne(padding_idx)
        positions = range_buf.expand_as(tensor)
        if left_pad:
            positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
        return positions * mask.long() + padding_idx * (1 - mask.long())

    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        make_positions.range_buf = torch.arange(padding_idx + 1, max_pos).type_as(tensor)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])