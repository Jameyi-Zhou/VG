from functools import partial
from .util.pos_embed import interpolate_pos_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .fusion_transformer import VisionTransformer
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights


class NormalWindow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, mask):
        pass


class ShuffleWindow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, mask):
        pass


class NormalWindowReverse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, mask):
        pass


class ShuffleWindowReverse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, mask):
        pass


class PatchMerge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, mask):
        return x, mask


class FusionViT(VisionTransformer):
    def __init__(self, **kwargs):
        super(FusionViT, self).__init__(**kwargs)
        
        self.img_size = kwargs['img_size']
        self.windowing = False
        self.num_patches = self.patch_embed.num_patches
        
        self.pm_layer1 = PatchMerge()
        self.pm_layer2 = PatchMerge()
        
        del self.norm
        del self.fc_norm
        del self.head_drop
        del self.head

    def normal_window(self, x, mask):
        bs, n, c = x.shape
        f_size = int((n - self.num_prefix_tokens - self.num_suffix_tokens) ** 0.5)
        w_size = f_size // 2
        w_bs = bs * 4

        cls_tokens = x[:, 0:1, :]
        v_tokens = x[:, 1:-1, :].reshape(bs, f_size, f_size, c)
        dstl_tokens = x[:, -1:, :]
        cls_mask = mask[:, 0:1]
        v_mask = mask[:, 1:-1].reshape(bs, f_size, f_size)
        dstl_mask = mask[:, -1:]

        cls_tokens = torch.repeat_interleave(cls_tokens, repeats=4, dim=0)
        v_tokens = v_tokens.view(bs, 2, w_size, 2, w_size, c).permute(0, 1, 3, 2, 4, 5).reshape(w_bs, w_size * w_size, c)
        dstl_tokens = torch.repeat_interleave(dstl_tokens, repeats=4, dim=0)

        cls_mask = torch.repeat_interleave(cls_mask, repeats=4, dim=0)
        v_mask = v_mask.view(bs, 2, w_size, 2, w_size).permute(0, 1, 3, 2, 4).reshape(w_bs, w_size * w_size)
        dstl_mask = torch.repeat_interleave(dstl_mask, repeats=4, dim=0)

        x = torch.cat([cls_tokens, v_tokens, dstl_tokens], dim=1)
        mask = torch.cat([cls_mask, v_mask, dstl_mask], dim=1)
        
        return x, mask

    def recover_from_normal_window(self, x, mask):
        w_bs, n, c = x.shape
        w_size = int((n - self.num_prefix_tokens - self.num_suffix_tokens) ** 0.5)
        f_size = w_size * 2
        bs = w_bs // 4

        cls_tokens = x[:, 0:1, :]
        v_tokens = x[:, 1:-1, :].reshape(w_bs, w_size, w_size, c)
        dstl_tokens = x[:, -1:, :]
        cls_mask = mask[:, 0:1]
        v_mask = mask[:, 1:-1].reshape(w_bs, w_size, w_size)
        dstl_mask = mask[:, -1:]
        
        cls_tokens = cls_tokens.reshape(bs, 4, c)
        cls_tokens = torch.mean(cls_tokens, dim=1, keepdim=True)
        v_tokens = v_tokens.view(bs, 2, 2, w_size, w_size, c).permute(0, 1, 3, 2, 4, 5).reshape(bs, f_size * f_size, c)        
        dstl_tokens = dstl_tokens.reshape(bs, 4, c)
        dstl_tokens = torch.mean(dstl_tokens, dim=1, keepdim=True)
        
        cls_mask = cls_mask[:bs]
        v_mask = v_mask.view(bs, 2, 2, w_size, w_size).permute(0, 1, 3, 2, 4).reshape(bs, f_size * f_size)
        dstl_mask = dstl_mask[:bs]

        x = torch.cat([cls_tokens, v_tokens, dstl_tokens], dim=1)
        mask = torch.cat([cls_mask, v_mask, dstl_mask], dim=1)

        return x, mask

    def shuffle_window(x, mask):
        pass

    def recover_from_shuffle_window(x, mask):
        pass

    def forward(self, tensor_list):
        x, m = tensor_list.decompose()
        x = self.patch_embed(x)
        bs, n, c = x.shape
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        dstl_tokens = self.dstl_token.expand(bs, -1, -1)
        x = torch.cat([cls_tokens, x, dstl_tokens], dim=1)
        x = x + self.pos_embed
        
        f_size = int(n ** 0.5)
        mask = F.interpolate(m[None].float(), size=(f_size, f_size)).to(torch.bool)[0].reshape(bs, -1)
        cls_mask = torch.zeros((bs, 1)).to(x.device).to(torch.bool)
        dstl_mask = torch.zeros((bs, 1)).to(x.device).to(torch.bool)        
        mask = torch.cat([cls_mask, mask, dstl_mask], dim=1)
        
        # x[bs, 402, 768], mask[bs, 402]
        if self.windowing:
            for i in range(0, 4):
                x, mask = self.normal_window(x, mask)
                x, attn = self.blocks[i](x, key_padding_mask=mask)
                x, mask = self.recover_from_normal_window(x, mask)
            self.pm_layer1(x, mask)  # patch_merge

            for i in range(4, 8):
                x, mask = self.normal_window(x, mask)
                x, attn = self.blocks[i](x, key_padding_mask=mask) 
                x, mask = self.recover_from_normal_window(x, mask)
            self.pm_layer2(x, mask)  # patch_merge

            for i in range(8, 12):
                x, mask = self.normal_window(x, mask)
                x, attn = self.blocks[i](x, key_padding_mask=mask)
                x, mask = self.recover_from_normal_window(x, mask)
        else:
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
    if args.use_mae:
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
    else:

        # 加载预训练的ViT模型
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.encoder.ln = torch.nn.Identity()
        model.heads = torch.nn.Identity()
    
    # print(msg)
    return model
    