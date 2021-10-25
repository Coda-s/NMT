import math

import torch
from torch import nn
from torch import nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.modules import MultiheadAttention

from .embedding_leca import (
    PositionalEmbedding,
    ConsPostionEmbedding,
    Embedding,
    LayerNorm,
    SinusoidalPositionalEmbedding,
    Linear
) 

class TransformerEncoderLeca(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, task=None):
        super().__init__(dictionary)
        self.dropout = args.dropout
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        if task is not None:
            self.src_dict = task.source_dictionary
            self.tgt_dict = task.target_dictionary
            
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx, 
            left_pad=args.left_pad_source,
            learned=args.encoder_learned_pos, 
        )
        # if not args.no_token_positional_embeddings else None
        
        if not hasattr(args, 'max_constraints_number'):
            args.max_constraints_number = 50 ## set the maximum constraints number to init segment embeddings 
        
        self.cons_pos_embed = ConsPostionEmbedding(args.decoder_embed_dim, self.padding_idx)
        self.seg_embed = Embedding(args.max_constraints_number, args.decoder_embed_dim, self.padding_idx)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)   
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths, decoder=None):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        """
        # print("find_unused_parameters: {}".format(self.find_unused_parameters))
        if decoder.sep_id not in src_tokens.view(-1):
            src_x = self.embed_scale * self.embed_tokens(src_tokens)
            src_posi_x = self.embed_positions(src_tokens)
            x = src_x + src_posi_x
        else:
            sep_position = min((src_tokens== decoder.sep_id).nonzero()[:,1])
            src_sent=src_tokens[:,:sep_position]

            src_x = self.embed_scale * self.embed_tokens(src_sent)
            src_posi_x = self.embed_positions(src_sent) 
            src_seg_emb=self.seg_embed(torch.zeros_like(src_sent))

            cons_sent = src_tokens[:,sep_position:] 
            cons_x =  self.embed_scale * decoder.embed_tokens(cons_sent)
            cons_posi_x = self.cons_pos_embed(cons_sent)
            seg_cons = torch.cumsum((cons_sent==decoder.sep_id),dim=1).type_as(cons_sent)
            seg_cons[(cons_sent==decoder.pad_id)] = torch.tensor([16]).type_as(seg_cons)        
            cons_seg_emb=self.seg_embed(seg_cons)

            x = torch.cat((src_x+src_posi_x+src_seg_emb, cons_x+cons_posi_x+cons_seg_emb), dim=1)
            # model_no_seg
            # x = torch.cat((src_x+src_posi_x, cons_x+cons_posi_x), dim=1)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers, layer.need_attn=True
        for layer in self.layers:            
            x, _ = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'src_tokens': src_tokens,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['src_tokens'] is not None:
            encoder_out['src_tokens'] = \
                encoder_out['src_tokens'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.need_attn=False 
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, attn = self.self_attn(query=x, key=x, value=x, \
            key_padding_mask=encoder_padding_mask, \
            need_weights=(not self.training and self.need_attn))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x,attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x



