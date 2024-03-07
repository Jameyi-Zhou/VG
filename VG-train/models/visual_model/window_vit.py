from functools import partial
from .util.pos_embed import interpolate_pos_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .vision_transformer import VisionTransformer
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights


# class NormalWindow(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, x, mask):
#         pass


# class ShuffleWindow(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, x, mask):
#         pass


# class NormalWindowReverse(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, x, mask):
#         pass


# class ShuffleWindowReverse(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, x, mask):
#         pass


class PatchMerging(nn.Module):
    def __init__(self, dim, num_extra_tokens, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_extra_tokens = num_extra_tokens
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, mask):
        bs, n, c = x.shape
        h = w = int((n - self.num_extra_tokens) ** 0.5)
        cls_tokens = x[:, 0:1, :]
        dstl_tokens = x[:, -1:, :]
        x = x[:, 1:-1, :].view(bs, h, w, c)
        
        x0 = x[:, 0::2, 0::2, :]  # bs h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # bs h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # bs h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # bs h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # bs h/2 w/2 4*c
        x = x.view(bs, -1, 4 * c)  # bs h/2*w/2 4*c
        x = self.norm(x)
        x = self.reduction(x)  # bs h/2*w/2 c
        x = torch.cat([cls_tokens, x, dstl_tokens], dim=1)

        cls_mask = mask[:, 0:1]
        dstl_mask = mask[:, -1:]
        mask = mask[:, 1:-1].view(bs, h, w)
        
        mask0 = mask[:, 0::2, 0::2]  # bs h/2 w/2
        mask1 = mask[:, 1::2, 0::2]  # bs h/2 w/2
        mask2 = mask[:, 0::2, 1::2]  # bs h/2 w/2
        mask3 = mask[:, 1::2, 1::2]  # bs h/2 w/2
        mask = (~(~mask0 + ~mask1 + ~mask2 + ~mask3)).view(bs, -1)
        mask = torch.cat([cls_mask, mask, dstl_mask], dim=1)

        return x, mask


class FusionViT(VisionTransformer):
    def __init__(self, windowing=True, shuffle=True, **kwargs):
        super(FusionViT, self).__init__(**kwargs)
        
        self.img_size = kwargs['img_size']
        self.windowing = windowing
        self.shuffle = shuffle
        self.num_patches = self.patch_embed.num_patches
        
        # self.pm_layer1 = PatchMerging(dim=self.embed_dim, num_extra_tokens=self.num_prefix_tokens+self.num_suffix_tokens)
        # self.pm_layer2 = PatchMerging(dim=self.embed_dim, num_extra_tokens=self.num_prefix_tokens+self.num_suffix_tokens)
        # self.resample0 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=(1, 1))
        # self.resample1 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=(1, 1))
        # self.resample2 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=(1, 1))
        # self.trans_conv1 = nn.ConvTranspose2d(in_channels=768, out_channels=768, kernel_size=4, stride=2, padding=1)
        # self.trans_conv2 = nn.ConvTranspose2d(in_channels=768, out_channels=768, kernel_size=4, stride=2, padding=1)
        self._reset_parameters()
        
        del self.norm
        del self.fc_norm
        del self.head_drop
        del self.head

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def shuffle_window(self, x, mask, idx):
        bs, n, c = x.shape
        n = n - self.num_prefix_tokens - self.num_suffix_tokens
        w_bs = bs * 4
        
        cls_tokens = x[:, 0:1, :]
        v_tokens = x[:, 1:-1, :]
        dstl_tokens = x[:, -1:, :]
        cls_mask = mask[:, 0:1]
        v_mask = mask[:, 1:-1]
        dstl_mask = mask[:, -1:]

        cls_tokens = torch.repeat_interleave(cls_tokens, repeats=4, dim=0)
        v_tokens = v_tokens[:, idx, :].view(bs, 4, n // 4, c).reshape(w_bs, n // 4, c)
        dstl_tokens = torch.repeat_interleave(dstl_tokens, repeats=4, dim=0)
        cls_mask = torch.repeat_interleave(cls_mask, repeats=4, dim=0)
        v_mask = v_mask[:, idx].view(bs, 4, n // 4).reshape(w_bs, n // 4)
        dstl_mask = torch.repeat_interleave(dstl_mask, repeats=4, dim=0)

        x = torch.cat([cls_tokens, v_tokens, dstl_tokens], dim=1)
        mask = torch.cat([cls_mask, v_mask, dstl_mask], dim=1)

        return x, mask

    def recover_from_shuffle_window(self, x, mask, idx):
        w_bs, n, c = x.shape
        n = n - self.num_prefix_tokens - self.num_suffix_tokens
        bs = w_bs // 4

        cls_tokens = x[:, 0:1, :]
        v_tokens = x[:, 1:-1, :]
        dstl_tokens = x[:, -1:, :]
        cls_mask = mask[:, 0:1]
        v_mask = mask[:, 1:-1]
        dstl_mask = mask[:, -1:]

        _, inverse_idx = idx.sort()
        cls_tokens = cls_tokens.reshape(bs, 4, c)
        cls_tokens = torch.mean(cls_tokens, dim=1, keepdim=True)
        v_tokens = v_tokens.view(bs, 4, n, c).reshape(bs, 4 * n, c)[:, inverse_idx, :]   
        dstl_tokens = dstl_tokens.reshape(bs, 4, c)
        dstl_tokens = torch.mean(dstl_tokens, dim=1, keepdim=True)
        cls_mask = cls_mask[:bs]
        v_mask = v_mask.view(bs, 4, n).reshape(bs, 4 * n)[:, inverse_idx]
        dstl_mask = dstl_mask[:bs]

        x = torch.cat([cls_tokens, v_tokens, dstl_tokens], dim=1)
        mask = torch.cat([cls_mask, v_mask, dstl_mask], dim=1)

        return x, mask

    def normal_window_forward(self, i, x, mask):
        x, mask = self.normal_window(x, mask)
        x, attn = self.blocks[i](x, key_padding_mask=mask)
        x, mask = self.recover_from_normal_window(x, mask)
        return x, mask, attn

    def shuffle_window_forward(self, i, x, mask, idx):
        x, mask = self.shuffle_window(x, mask, idx)
        x, attn = self.blocks[i](x, key_padding_mask=mask)
        x, mask = self.recover_from_shuffle_window(x, mask, idx)
        return x, mask, attn

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
        # import pdb;pdb.set_trace()
        output_x = []
        output_mask = []
        # x[bs, 402, 768], mask[bs, 402]
        
        if self.windowing:
            
            if self.train() and self.shuffle:
                for i in range(0, 2):
                    idx0 = torch.randperm(n)
                    x, mask, attn = self.normal_window_forward(i * 2, x, mask)
                    x, mask, attn = self.shuffle_window_forward(i * 2 + 1, x, mask, idx0)
            else:
                for i in range(0, 4):
                    x, mask, attn = self.normal_window_forward(i, x, mask)
            output_x.append(x[:, 1:-1, :])
            output_mask.append(mask[:, 1:-1])
            # x, mask = self.pm_layer1(x, mask)  # patch merging

            
            if self.train() and self.shuffle:
                for i in range(2, 4):
                    idx1 = torch.randperm(n // 4)
                    x, mask, attn = self.normal_window_forward(i * 2, x, mask)
                    x, mask, attn = self.shuffle_window_forward(i * 2 + 1, x, mask, idx1)
            else:
                for i in range(4, 8):
                    x, mask, attn = self.normal_window_forward(i, x, mask)
            output_x.append(x[:, 1:-1, :])
            output_mask.append(mask[:, 1:-1])
            x, mask = self.pm_layer2(x, mask)  # patch merging
 
            for i in range(8, 12):
                x, attn = self.blocks[i](x, key_padding_mask=mask)
            cls_tokens, dstl_tokens = x[:, :1, :], x[:, -1:, :]
            cls_mask, dstl_mask = mask[:, :1], mask[:, -1:]
            output_x.append(x[:, 1:-1, :])
            output_mask.append(mask[:, 1:-1])
            
            '''
            ' @ablation study 使用反卷积上采样
            '''
            # f_size0 = f_size
            # f_size1 = f_size0 // 2
            # f_size2 = f_size1 // 2
            # x2 = output_x[2].view(bs, f_size2, f_size2, -1).permute(0, 3, 1, 2)
            # x1 = output_x[1].view(bs, f_size1, f_size1, -1).permute(0, 3, 1, 2)
            # x0 = output_x[0].view(bs, f_size0, f_size0, -1).permute(0, 3, 1, 2)
            # x2 = x2
            # x2_unsample = self.trans_conv1(x2)
            # x1 = x1 + x2_unsample
            # x1_unsample = self.trans_conv1(x1)
            # x0 = x0 + x1_unsample
            '''
            ' @ablation study 使用双三次插值上采样
            '''
            # x2 = x2
            # x2_unsample = torch.nn.functional.interpolate(x2, size=(f_size1, f_size1), mode='bicubic', align_corners=False)
            # x1 = x1 + x2_unsample
            # x1_unsample = torch.nn.functional.interpolate(x1, size=(f_size0, f_size0), mode='bicubic', align_corners=False)
            # x0 = x0 + x1_unsample
            
            '''
            ' concat
            '''
            # x2 = x2.permute(0, 2, 3, 1).view(bs, f_size2*f_size2, -1)
            # x1 = x1.permute(0, 2, 3, 1).view(bs, f_size1*f_size1, -1)
            # x0 = x0.permute(0, 2, 3, 1).view(bs, f_size0*f_size0, -1)
            # x_list = [cls_tokens, x0, x1, x2, dstl_tokens]
            # mask_list = [cls_mask, output_mask[0], output_mask[1], output_mask[2], dstl_mask]
            return x_list, mask_list
        else:
            for i in range(0, 12):
                x, attn = self.blocks[i](x, key_padding_mask=mask)
            return x, mask

        # attn_map = attn[0, :, 0 , 1:]
        # attn_map_norm = attn_map / attn_map.sum(dim=1, keepdim=True)
        # attn_map_norm = attn_map_norm.reshape(12, 20, -1)

        # fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        # for i, ax in enumerate(axes.flatten()):
        #     # 归一化注意力权重
        #     sns.heatmap(attn_map_norm[i].detach().cpu().numpy(), ax=ax, cmap='viridis')
        #     ax.set_title(f'Head {i+1}')
        
        # plt.savefig('heatmap.png')


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
    