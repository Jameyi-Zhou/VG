from functools import partial
from .util.pos_embed import interpolate_pos_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .fusion_transformer import VisionTransformer
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


class FusionViT(VisionTransformer):
    def __init__(self, **kwargs):
        super(FusionViT, self).__init__(**kwargs)
        self.img_size=kwargs['img_size']
        self.num_patches = self.patch_embed.num_patches
        self.num_scales = 3
        
        self.conv_downsample1 = DownSampleLayer(self.embed_dim)
        self.conv_downsample2 = DownSampleLayer(self.embed_dim)
        self.scale_pos_embed = nn.Embedding(self.num_scales, self.embed_dim)
        self.cls_mask = torch.zeros((1, 1))

        del self.norm
        del self.fc_norm
        del self.head_drop
        del self.head

    def forward(self, input_tensors):
        x, m = input_tensors.decompose()
        bs = x.shape[0]

        cls_tokens = self.cls_token.unsqueeze(1).expand(bs, 21, -1, -1)
        x0 = self.patch_embed(x)  # (bs, 768, 40, 40)
        x1 = self.conv_downsample1(x0)  # (bs, 768, 20, 20)
        x2 = self.conv_downsample2(x1)  # (bs, 768, 10, 10)
        
        # applying different scale mask to different scale fetures 
        x_mask0 = F.interpolate(m[None].float(), size=(x0.shape[-1], x0.shape[-1])).to(torch.bool)[0]
        x_mask0 = x_mask0.unfold(1, 10, 10).unfold(2, 10, 10)
        x_mask0 = x_mask0.contiguous().view(bs, 16, 10, 10).flatten(2)
        x_mask1 = F.interpolate(m[None].float(), size=(x1.shape[-1], x1.shape[-1])).to(torch.bool)[0]
        x_mask1 = x_mask1.unfold(1, 10, 10).unfold(2, 10, 10)
        x_mask1 = x_mask1.contiguous().view(bs, 4, 10, 10).flatten(2)
        x_mask2 = F.interpolate(m[None].float(), size=(x2.shape[-1], x2.shape[-1])).to(torch.bool)[0]
        x_mask2 = x_mask2.unfold(1, 10, 10).unfold(2, 10, 10)
        x_mask2 = x_mask2.contiguous().view(bs, 1, 10, 10).flatten(2)
        self.cls_mask = self.cls_mask.to(x.device).to(torch.bool)
        cls_mask = self.cls_mask.unsqueeze(0).expand(bs, 21, -1)
        mask = torch.cat([x_mask0, x_mask1, x_mask2], dim=1)
        mask = torch.cat([cls_mask, mask], dim=2)
        scale_pos_embed0 = self.scale_pos_embed.weight[0:1].unsqueeze(0).unsqueeze(0).expand(-1, 16, 101, -1)
        scale_pos_embed1 = self.scale_pos_embed.weight[1:2].unsqueeze(0).unsqueeze(0).expand(-1, 4, 101, -1)
        scale_pos_embed2 = self.scale_pos_embed.weight[2:3].unsqueeze(0).unsqueeze(0).expand(-1, 1, 101, -1)
        scale_pos_embed = torch.cat([scale_pos_embed0, scale_pos_embed1, scale_pos_embed2], dim=1)
        x0 = x0.unfold(2, 10, 10).unfold(3, 10, 10)
        x0 = x0.contiguous().view(bs, 768, 16, 10, 10).flatten(3).permute(0, 2, 3, 1)
        x1 = x1.unfold(2, 10, 10).unfold(3, 10, 10)
        x1 = x1.contiguous().view(bs, 768, 4, 10, 10).flatten(3).permute(0, 2, 3, 1)
        x2 = x2.unfold(2, 10, 10).unfold(3, 10, 10)
        x2 = x2.contiguous().view(bs, 768, 1, 10, 10).flatten(3).permute(0, 2, 3, 1)
        x = torch.cat([x0, x1, x2], dim=1)
        x = torch.cat([cls_tokens, x], dim=2)
        x = x + self.pos_embed + scale_pos_embed
        # x = self.pos_drop(x)
        for i in range(0, 12):
            x = self.blocks[i](x, mask)
        
        # attn_map = attn[0, :, 0 , 1:]
        # attn_map_norm = attn_map / attn_map.sum(dim=1, keepdim=True)
        # attn_map_norm = attn_map_norm.reshape(12, 20, -1)

        # fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        # for i, ax in enumerate(axes.flatten()):
        #     # 归一化注意力权重
        #     sns.heatmap(attn_map_norm[i].detach().cpu().numpy(), ax=ax, cmap='viridis')
        #     ax.set_title(f'Head {i+1}')
        
        # plt.savefig('heatmap.png')

        return x, mask
    

def vit_base_patch16(**kwargs):
    model = FusionViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = FusionViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = FusionViT(
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
    model.load_state_dict(checkpoint_model, strict=False)
    # load pre-trained model
    
    # print(msg)
    return model
    