import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.fusion_vit import build_vit
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy


class VGModel(nn.Module):
    def __init__(self, args):
        super(VGModel, self).__init__()
        
        self.num_visu_token = int((args.imsize / 64) ** 2) + int((args.imsize / 32) ** 2) + int((args.imsize / 16) ** 2) + 1  # cls_token
        self.num_text_token = args.max_query_len

        self.visumodel = build_vit(args)  # 构造vit分支
        self.textmodel = build_bert(args)  # 构造bert分支
        hidden_dim = args.vl_hidden_dim

        num_total = self.num_visu_token + self.num_text_token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.embed_dim, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        # language branch
        text_src, text_mask = self.textmodel(text_data)
        assert text_mask is not None
        
        # visual branch
        visu_src, visu_mask = self.visumodel(img_data)
        assert visu_mask is not None

        visu_src = torch.cat([self.visumodel.cls_token.expand(bs, -1, -1), visu_src[:, :, 0:100, :].reshape(bs, 2100, -1)], dim=1)
        visu_mask = torch.cat([self.visumodel.cls_mask.expand(bs, -1), visu_mask[:, :, 0:100].reshape(bs, 2100)], dim=1)
        text_src = self.text_proj(text_src).permute(1, 0, 2) # (N*B)xC
        visu_src = self.visu_proj(visu_src).permute(1, 0, 2) # (N*B)xC


        vl_src = torch.cat([visu_src, text_src], dim=0)
        vl_mask = torch.cat([visu_mask, text_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).expand(-1, bs, -1)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        
        # bbox回归
        vg_hs = vg_hs[0]
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
