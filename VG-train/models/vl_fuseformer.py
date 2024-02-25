# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# class FuseSparseAttention(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             qkv_bias: bool = False,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.,
#             norm_layer: nn.Module = nn.LayerNorm,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.norm = norm_layer(dim)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens) -> torch.Tensor:
#         B, _, C = visu_tokens.shape
#         z_i, z_t, z_d = visu_tokens, text_tokens, dual_query_tokens
#         N_i, N_t, N_d = z_i.shape[1], z_t.shape[1], z_d.shape[1]
#         qkv_i = self.qkv(z_i).reshape(B, N_i, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         qkv_t = self.qkv(z_t).reshape(B, N_t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         qkv_d = self.qkv(z_d).reshape(B, N_d, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q_i, k_i, v_i = qkv_i.unbind(0)
#         q_t, k_t, v_t = qkv_t.unbind(0)
#         q_d, k_d, v_d = qkv_d.unbind(0)
        
#         # import pdb; pdb.set_trace()
#         q_d = q_d * self.scale
#         attn = q_d @ k_d.transpose(-2, -1)
#         attn = attn.softmax(dim=-1)
#         q = (attn @ v_d).transpose(1, 2).reshape(B, N_d, C)
#         q = self.norm(q + z_d)
#         q1 = q.reshape(B, N_d, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # z_d 或 q_d ?
#         # visual query cross attention
#         i_mask = visu_mask
#         attn_d2i = q1 @ k_i.transpose(-2, -1)
#         attn_d2i_mask = i_mask.unsqueeze(1).expand(-1, N_d, -1)
#         attn_d2i = attn_d2i.masked_fill(attn_d2i_mask.unsqueeze(1), float('-inf'))
#         attn_d2i = attn_d2i.softmax(dim=-1)
#         z_i = (attn_d2i @ v_i).transpose(1, 2).reshape(B, N_d, C)
#         # text query cross attention
#         t_mask = text_mask
#         attn_d2t = q1 @ k_t.transpose(-2, -1)
#         attn_d2t_mask = t_mask.unsqueeze(1).expand(-1, N_d, -1)
#         attn_d2t = attn_d2t.masked_fill(attn_d2t_mask.unsqueeze(1), float('-inf'))
#         attn_d2t = attn_d2t.softmax(dim=-1)
#         z_t = (attn_d2t @ v_t).transpose(1, 2).reshape(B, N_d, C)
        
#         x = z_i + z_t + q

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
    
#         # attn_i2t = q_i @ k_t.transpose(-2, -1)
#         # attn_t2i = q_t @ k_i.transpose(-2, -1)
        
#         # attn_i2t_mask = t_mask.unsqueeze(1).expand(-1, num_visu_token, -1)
#         # attn_i2t = attn_i2t.masked_fill(attn_i2t_mask.unsqueeze(1), float('-inf'))
#         # attn_i2t = attn_i2t.softmax(dim=-1)

#         # attn_t2i_mask = i_mask.unsqueeze(1).expand(-1, num_text_token, -1)
#         # attn_t2i = attn_t2i.masked_fill(attn_t2i_mask.unsqueeze(1), float('-inf'))
#         # attn_t2i = attn_t2i.softmax(dim=-1)

#         # attn_i2t = attn_i2t.masked_fill(attn_t2i_mask.transpose(1, 2).unsqueeze(1), 0)
#         # attn_t2i = attn_t2i.masked_fill(attn_i2t_mask.transpose(1, 2).unsqueeze(1), 0)
#         # v_t2i = attn_t2i @ v_i
#         # x_i = attn_i2t @ v_t2i
#         # x_t = attn_t2i @ v_i


# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=True):
#         super().__init__()
#         self.fsattn = FuseSparseAttention(d_model, nhead)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#     def forward_post(self, visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens):
#         src = dual_query_tokens
#         src1 = self.fsattn(visu_tokens, text_tokens, visu_mask, text_mask, src)  # FuseSparseAttention
#         src = src + self.dropout1(src1)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # FFN
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

#     def forward_pre(self, visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens):
#         src = dual_query_tokens
#         src1 = self.norm1(src)
#         src1 = self.fsattn(visu_tokens, text_tokens, visu_mask, text_mask, src1)  # FuseSparseAttention
#         src = src + self.dropout1(src1)
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))  # FFN
#         src = src + self.dropout2(src2)
#         return src

#     def forward(self, visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens):
#         if self.normalize_before:
#             return self.forward_pre(visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens)
#         return self.forward_post(visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens)


# class TransformerEncoder(nn.Module):

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens):

#         for layer in self.layers:
#             dual_query_tokens = layer(visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens)
#         output = dual_query_tokens
#         if self.norm is not None:
#             output = self.norm(output)

#         return output
    

# class FuseFormer(nn.Module):

#     def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", num_dual_query_tokens=256, normalize_before=True):
#         super().__init__()

#         encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
#                                                 dropout, activation, normalize_before)
#         encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
#         self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
#         self.dual_query = nn.Embedding(num_dual_query_tokens, d_model)

#         self._reset_parameters()

#         self.d_model = d_model
#         self.nhead = nhead

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, cls_tokens, visu_tokens, text_tokens, visu_mask, text_mask, num_visu_token) :
#         bs = visu_tokens.shape[0]
#         dual_query_tokens = self.dual_query.weight.unsqueeze(0).expand(bs, -1, -1)
#         dual_query_tokens = torch.cat([cls_tokens, dual_query_tokens], dim=1)
#         return self.encoder(visu_tokens, text_tokens, visu_mask, text_mask, dual_query_tokens)


# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# def build_vl_fuseformer(args):
#     return FuseFormer(
#         d_model=args.vl_hidden_dim,
#         dropout=args.vl_dropout,
#         nhead=args.vl_nheads,
#         dim_feedforward=args.vl_dim_feedforward,
#         num_encoder_layers=args.vl_enc_layers,
#         num_dual_query_tokens=args.num_dual_query_tokens,
#         normalize_before=True,
#         activation="gelu"
#     )


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class VisionLanguageEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed=None):
        return self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_vl_fuseformer(args):
    return VisionLanguageEncoder(
        d_model=args.vl_hidden_dim,
        dropout=args.vl_dropout,
        nhead=args.vl_nheads,
        dim_feedforward=args.vl_dim_feedforward,
        num_encoder_layers=args.vl_enc_layers,
        normalize_before=True,
        activation="gelu"
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")