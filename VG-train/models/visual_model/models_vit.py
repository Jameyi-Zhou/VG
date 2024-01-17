from functools import partial
from .util.pos_embed import interpolate_pos_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .vision_transformer import VisionTransformer
from PIL import Image


class DownSampleLayer(nn.Module):
    def __init__(self, in_chans) -> None:
        super().__init__()
        mid_chans = 256
        self.conv0 = nn.Conv2d(in_chans, mid_chans, kernel_size=(1, 1))
        self.conv1 = nn.Conv2d(mid_chans, mid_chans, kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.conv2 = nn.Conv2d(mid_chans, in_chans, kernel_size=(1, 1))
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ViT(VisionTransformer):
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)
        self.img_size=kwargs['img_size']
        self.num_patches = self.patch_embed.num_patches
        self.num_scales = 3
        
        self.conv_downsample1 = DownSampleLayer(self.embed_dim)
        self.conv_downsample2 = DownSampleLayer(self.embed_dim)
        self.scale_pos_embed = nn.Embedding(self.num_scales, self.embed_dim)

        del self.norm
        del self.fc_norm
        del self.head_drop
        del self.head

    def forward(self, input_tensors, text_list):
        x, m = input_tensors.decompose()
        bs = x.shape[0]

        cls_tokens = self.cls_token.expand(bs, -1, -1)
        x0 = self.patch_embed(x)  # (bs, 768, 40, 40)
        x1 = self.conv_downsample1(x0)
        x2 = self.conv_downsample2(x1)
        
        # applying different scale mask to different scale fetures 
        v_mask0 = F.interpolate(m[None].float(), size=(x0.shape[-1], x0.shape[-1])).to(torch.bool)[0].flatten(1)
        v_mask1 = F.interpolate(m[None].float(), size=(x1.shape[-1], x1.shape[-1])).to(torch.bool)[0].flatten(1)
        v_mask2 = F.interpolate(m[None].float(), size=(x2.shape[-1], x2.shape[-1])).to(torch.bool)[0].flatten(1)
        cls_mask = torch.zeros((bs, 1)).to(x.device).to(torch.bool)
        x_mask = torch.cat([cls_mask, v_mask0, v_mask1, v_mask2], dim=1)

        x0 = x0.flatten(2).transpose(1, 2)
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        scale_pos_embed = self.scale_pos_embed.weight.unsqueeze(0)
        x0 = x0 + scale_pos_embed[:, 0:1, :].expand(-1, x0.shape[-2], -1) + self.pos_embed[:, 0:x0.shape[-2], :]
        x1 = x1 + scale_pos_embed[:, 1:2, :].expand(-1, x1.shape[-2], -1) + self.pos_embed[:, x0.shape[-2]:x1.shape[-2]+x0.shape[-2], :]
        x2 = x2 + scale_pos_embed[:, 2:3, :].expand(-1, x2.shape[-2], -1) + self.pos_embed[:, x1.shape[-2]+x0.shape[-2]:x2.shape[-2]+x1.shape[-2]+x0.shape[-2], :]
        x = torch.cat([cls_tokens, x0, x1, x2], dim=1)

        t = text_list[0]
        t_mask = text_list[1]
        # x = self.pos_drop(x)
        import pdb;pdb.set_trace()
        
        for i in range(0, 12):
            x = self.blocks[i](x, x_mask, t, t_mask)
        
        # attn_map = attn[0, :, 0 , 1:]
        # attn_map_norm = attn_map / attn_map.sum(dim=1, keepdim=True)
        # attn_map_norm = attn_map_norm.reshape(12, 20, -1)

        # fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        # for i, ax in enumerate(axes.flatten()):
        #     # 归一化注意力权重
        #     sns.heatmap(attn_map_norm[i].detach().cpu().numpy(), ax=ax, cmap='viridis')
        #     ax.set_title(f'Head {i+1}')
        
        # plt.savefig('heatmap.png')

        return x, x_mask
    

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
    