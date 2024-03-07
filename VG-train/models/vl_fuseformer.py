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


class SparseFuseAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # self.k_x0_sparse = nn.Linear(1600, 200)
        # self.v_x0_sparse = nn.Linear(1600, 200)
        # self.k_x1_sparse = nn.Linear(400, 200)
        # self.v_x1_sparse = nn.Linear(400, 200)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, src, visu_mask, text_mask) -> torch.Tensor:
        B, N, C = src.shape
        N = src.shape[1]
        N_t = text_mask.shape[1]

        qkv = self.qkv(src).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q_cls, q_x0, q_x1, q_x2, q_t  = q[:, :, :1, :], q[:, :, 1:1601, :], q[:, :, 1601:2001, :], q[:, :, 2001:2101, :], q[:, :, 2101:2121, :]
        k_cls, k_x0, k_x1, k_x2, k_t  = k[:, :, :1, :], k[:, :, 1:1601, :], k[:, :, 1601:2001, :], k[:, :, 2001:2101, :], k[:, :, 2101:2121, :]
        v_cls, v_x0, v_x1, v_x2, v_t  = v[:, :, :1, :], v[:, :, 1:1601, :], v[:, :, 1601:2001, :], v[:, :, 2001:2101, :], v[:, :, 2101:2121, :]
        
        visu_mask = visu_mask.unsqueeze(1)
        text_mask = text_mask.unsqueeze(1)
        cls_mask, x0_mask, x1_mask, x2_mask = visu_mask[:, :, :1], visu_mask[:, :, 1:1601], visu_mask[:, :, 1601:2001], visu_mask[:, :, 2001:2101]

        

        # cls forward
        q_cls = q_cls * self.scale
        attn = q_cls @ (k).transpose(-2, -1)
        attn_mask = torch.cat([visu_mask, text_mask], dim=2)
        attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn = attn.softmax(dim=-1)
        cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        
        # x2 forward
        q_x2 = q_x2 * self.scale
        attn = q_x2 @ torch.cat([k_x0, k_x1, k_x2, k_t], dim=2).transpose(-2, -1)
        # attn_mask
        attn_mask = torch.cat([x0_mask, x1_mask, x2_mask, text_mask], dim=2).expand(-1, 100, -1)
        attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn = attn.softmax(dim=-1)
        x2 = (attn @ torch.cat([v_x0, v_x1, v_x2, v_t], dim=2)).transpose(1, 2).reshape(B, 100, C)
        
        # x1 forward
        q_x1 = q_x1 * self.scale
        attn = q_x1 @ torch.cat([k_x0, k_x1, k_t], dim=2).transpose(-2, -1)
        # attn_mask 
        attn_mask = torch.cat([x0_mask, x1_mask, text_mask], dim=2).expand(-1, 400, -1)
        attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn = attn.softmax(dim=-1)
        x1 = (attn @ torch.cat([v_x0, v_x1, v_t], dim=2)).transpose(1, 2).reshape(B, 400, C)

        # x0 forward
        q_x0 = q_x0 * self.scale
        attn = q_x0 @ torch.cat([k_x0, k_t], dim=2).transpose(-2, -1)
        # @attn_mask 
        attn_mask = torch.cat([x0_mask, text_mask], dim=2).expand(-1, 1600, -1)
        attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn = attn.softmax(dim=-1)
        x0 = (attn @ torch.cat([v_x0, v_t], dim=2)).transpose(1, 2).reshape(B, 1600, C)

        # t forward
        q_t = q_t * self.scale
        attn = q_t @ (k).transpose(-2, -1)
        attn_mask = torch.cat([visu_mask, text_mask], dim=2).expand(-1, N_t, -1)
        attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn = attn.softmax(dim=-1)
        t = (attn @ v).transpose(1, 2).reshape(B, N_t, C)

        x = torch.cat([cls, x0, x1, x2, t], dim=1) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
        # attn_i2t = q_i @ k_t.transpose(-2, -1)
        # attn_t2i = q_t @ k_i.transpose(-2, -1)
        
        # attn_i2t_mask = t_mask.unsqueeze(1).expand(-1, num_visu_token, -1)
        # attn_i2t = attn_i2t.masked_fill(attn_i2t_mask.unsqueeze(1), float('-inf'))
        # attn_i2t = attn_i2t.softmax(dim=-1)

        # attn_t2i_mask = i_mask.unsqueeze(1).expand(-1, num_text_token, -1)
        # attn_t2i = attn_t2i.masked_fill(attn_t2i_mask.unsqueeze(1), float('-inf'))
        # attn_t2i = attn_t2i.softmax(dim=-1)

        # attn_i2t = attn_i2t.masked_fill(attn_t2i_mask.transpose(1, 2).unsqueeze(1), 0)
        # attn_t2i = attn_t2i.masked_fill(attn_i2t_mask.transpose(1, 2).unsqueeze(1), 0)
        # v_t2i = attn_t2i @ v_i
        # x_i = attn_i2t @ v_t2i
        # x_t = attn_t2i @ v_i


class SparseFuseLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.fsattn = SparseFuseAttention(d_model, nhead)
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

    def forward_post(self, src, visu_mask, text_mask):
        src1 = self.fsattn(src, visu_mask, text_mask)  # FuseSparseAttention
        src = src + self.dropout1(src1)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # FFN
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src , visu_mask, text_mask):
        src1 = self.norm1(src)
        src1 = self.fsattn(src1, visu_mask, text_mask)  # FuseSparseAttention
        src = src + self.dropout1(src1)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))  # FFN
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, visu_mask, text_mask):
        if self.normalize_before:
            return self.forward_pre(src, visu_mask, text_mask)
        return self.forward_post(src, visu_mask, text_mask)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, visu_mask, text_mask):

        for layer in self.layers:
            src = layer(src, visu_mask, text_mask)
        output = src
        if self.norm is not None:
            output = self.norm(output)

        return output
    

class FuseFormer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", num_dual_query_tokens=256, normalize_before=True):
        super().__init__()

        encoder_layer = SparseFuseLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.dual_query = nn.Embedding(num_dual_query_tokens, d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, cls_tokens, visu_tokens, text_tokens, visu_mask, text_mask) :
        src = torch.cat([cls_tokens, visu_tokens, text_tokens], dim=1)
        # bs = visu_tokens.shape[0]
        # dual_query_tokens = self.dual_query.weight.unsqueeze(0).expand(bs, -1, -1)
        # dual_query_tokens = torch.cat([cls_tokens, dual_query_tokens], dim=1)
        return self.encoder(src, visu_mask, text_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_vl_fuseformer(args):
    return FuseFormer(
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


# class VisionLanguageEncoder(nn.Module):

#     def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
#                  num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()

#         encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
#                                                 dropout, activation, normalize_before)
#         encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
#         self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

#         self._reset_parameters()

#         self.d_model = d_model
#         self.nhead = nhead

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, src, mask, pos_embed=None):
#         return self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)


# class TransformerEncoder(nn.Module):

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src,
#                 mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         output = src

#         for layer in self.layers:
#             output = layer(output, src_mask=mask,
#                            src_key_padding_mask=src_key_padding_mask, pos=pos)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output


# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self,
#                      src,
#                      src_mask: Optional[Tensor] = None,
#                      src_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None):
#         q = k = self.with_pos_embed(src, pos)
#         src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

#     def forward_pre(self, src,
#                     src_mask: Optional[Tensor] = None,
#                     src_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None):
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#         src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         src = src + self.dropout2(src2)
#         return src

#     def forward(self, src,
#                 src_mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
#         return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# def build_vl_fuseformer(args):
#     return VisionLanguageEncoder(
#         d_model=args.vl_hidden_dim,
#         dropout=args.vl_dropout,
#         nhead=args.vl_nheads,
#         dim_feedforward=args.vl_dim_feedforward,
#         num_encoder_layers=args.vl_enc_layers,
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