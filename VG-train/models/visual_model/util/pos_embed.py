# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size1 = int(num_patches ** 0.5)
        new_size2 = int(num_patches ** 0.5 / 2)
        new_size3 = int(num_patches ** 0.5 / 4)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size1:
            print("Position interpolate from %dx%d to %dx%d, %dx%d, %dx%d" % (orig_size, orig_size, new_size1, new_size1, 
                                                                              new_size2, new_size2, new_size3, new_size3))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens1 = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size1, new_size1), mode='bicubic', align_corners=False)
            pos_tokens1 = pos_tokens1.unfold(2, 10, 10).unfold(3, 10, 10)
            pos_tokens1 = pos_tokens1.contiguous().view(1, 768, 16, 10, 10).flatten(3)
            pos_tokens2 = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size2, new_size2), mode='bicubic', align_corners=False)
            pos_tokens2 = pos_tokens2.unfold(2, 10, 10).unfold(3, 10, 10)
            pos_tokens2 = pos_tokens2.contiguous().view(1, 768, 4, 10, 10).flatten(3)
            pos_tokens3 = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size3, new_size3), mode='bicubic', align_corners=False)
            pos_tokens3 = pos_tokens3.unfold(2, 10, 10).unfold(3, 10, 10)
            pos_tokens3 = pos_tokens3.contiguous().view(1, 768, 1, 10, 10).flatten(3)
            extra_tokens = extra_tokens.unsqueeze(1).expand(-1, 21, -1, -1)
            new_pos_embed = torch.cat([pos_tokens1, pos_tokens2, pos_tokens3], dim=2).permute(0, 2, 3, 1)
            new_pos_embed = torch.cat([extra_tokens, new_pos_embed], dim=2)
            checkpoint_model['pos_embed'] = new_pos_embed
