import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_model.fusion_vit import build_vit
from .language_model.model_bert import build_bert
from .vl_fuseformer import build_vl_fuseformer
from .language_model.model_roberta import build_roberta
from utils.box_utils import xywh2xyxy


class VGModel(nn.Module):
    def __init__(self, args):
        super(VGModel, self).__init__()
        '''
        ' @TODO:不同尺寸的输入序列选择不同的FFN hidden_dim
        '''
        self.num_visu_token0 = int((args.imsize / 16) ** 2)
        self.num_visu_token1 = int((args.imsize / 32) ** 2)
        self.num_visu_token2 = int((args.imsize / 64) ** 2)
        self.num_visu_token = self.num_visu_token0 + self.num_visu_token1 + self.num_visu_token2 + 1  # cls_token
        self.num_text_token = args.max_query_len

        self.visumodel = build_vit(args)  # 构造vit分支
        self.textmodel = build_bert(args)  # 构造bert分支
        
        hidden_dim = args.vl_hidden_dim
        num_total = self.num_visu_token + self.num_text_token
        
        self.text_pos_embed = nn.Embedding(self.num_text_token, hidden_dim)

        self.extra_pos_embed = nn.Embedding(1, hidden_dim)  # cls_token
        self.visu_scale_pos_embed = nn.Embedding(3, hidden_dim)  # different scales
        self.visu_pos_embed0 = nn.Embedding(self.num_visu_token0, hidden_dim)
        self.visu_pos_embed1 = nn.Embedding(self.num_visu_token1, hidden_dim)
        self.visu_pos_embed2 = nn.Embedding(self.num_visu_token2, hidden_dim)
        
        if args.use_mae:
            self.visu_proj = nn.Linear(self.visumodel.embed_dim, hidden_dim)
        else:
            self.visu_proj = nn.Linear(self.visumodel.hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_fuseformer = build_vl_fuseformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def text_tokens_with_pos_embed(self, text_src):
        text_src = text_src + self.text_pos_embed.weight.unsqueeze(0)
        return text_src

    def visu_tokens_with_pos_embed(self, list):
        cls_tokens, x0, x1, x2, _ = list
        cls_tokens = cls_tokens + self.extra_pos_embed.weight[:1, :].unsqueeze(0)
        x0 = x0 + self.visu_scale_pos_embed.weight[0:1, :].expand(self.num_visu_token0, -1).unsqueeze(0)  # scale position embedding
        x0 = x0 + self.visu_pos_embed0.weight.unsqueeze(0)  # token position embedding
        x1 = x1 + self.visu_scale_pos_embed.weight[1:2, :].expand(self.num_visu_token1, -1).unsqueeze(0)
        x1 = x1 + self.visu_pos_embed1.weight.unsqueeze(0)
        x2 = x2 + self.visu_scale_pos_embed.weight[2:3, :].expand(self.num_visu_token2, -1).unsqueeze(0)
        x2 = x2 + self.visu_pos_embed2.weight.unsqueeze(0)
        return cls_tokens, torch.cat([x0, x1, x2], dim=1)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        
        # language branch
        text_src, text_mask = self.textmodel(text_data)
        text_src = self.text_proj(text_src)
        text_tokens = self.text_tokens_with_pos_embed(text_src)
        # visual branch
        visu_src_list, visu_mask_list = self.visumodel(img_data)
        for i in range(len(visu_src_list)):
            visu_src_list[i] = self.visu_proj(visu_src_list[i])
        cls_tokens, visu_tokens = self.visu_tokens_with_pos_embed(visu_src_list)
        cls_mask, mask0, mask1, mask2, _ = visu_mask_list
        visu_mask = torch.cat([mask0, mask1, mask2], dim=1)
        
        ####test
        vl_src = torch.cat([cls_tokens, visu_tokens[:, 2000:, :], text_tokens], dim=1).permute(1, 0, 2)
        vl_mask = torch.cat([cls_mask, mask2, text_mask], dim=1)
        vg_hs = self.vl_fuseformer(vl_src, vl_mask)[0]
        ####
        
        # vg_hs = self.vl_fuseformer(cls_tokens, visu_tokens, text_tokens, visu_mask, text_mask, self.num_visu_token)  # fusion
        # vg_hs = vg_hs.permute(1, 0, 2)[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()
        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
