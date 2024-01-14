from functools import partial
from .util.pos_embed import interpolate_pos_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .vision_transformer import VisionTransformer
from PIL import Image


class ViT(VisionTransformer):
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)
        self.img_size=kwargs['img_size']
        self.num_patches = self.patch_embed.num_patches
        del self.norm
        del self.fc_norm
        del self.head_drop
        del self.head


    def forward(self, tensor_list):
        x, m = tensor_list.decompose()
        bs = x.shape[0]
        v_mask = F.interpolate(m[None].float(), size=(self.img_size // 16, self.img_size // 16)).to(torch.bool)[0]
        cls_mask = torch.zeros((bs, 1)).to(x.device).to(torch.bool)
        v_mask = v_mask.flatten(1)
        mask = torch.cat([cls_mask, v_mask], dim=1)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for i in range(0, 12):
            x, attn = self.blocks[i](x, key_padding_mask=mask)
        
        # attn_map = attn[0, :, 0 , 1:]
        # attn_map_norm = attn_map / attn_map.sum(dim=1, keepdim=True)
        # attn_map_norm = attn_map_norm.reshape(12, 20, -1)

        # fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        # for i, ax in enumerate(axes.flatten()):
        #     # 归一化注意力权重
        #     sns.heatmap(attn_map_norm[i].detach().cpu().numpy(), ax=ax, cmap='viridis')
        #     ax.set_title(f'Head {i+1}')
        
        # plt.savefig('heatmap.png')
        # import pdb;pdb.set_trace()

        return x, mask  # [b, 401, 768]
    

def vit_base_patch16(**kwargs):
    model = ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = ViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = ViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def build_vit(args, **kwargs):
    model = vit_base_patch16(img_size=args.imsize)
    checkpoint = torch.load(args.vit_checkpoint, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.vit_checkpoint)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    return model
    