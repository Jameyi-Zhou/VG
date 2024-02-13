from .modeling_roberta import RobertaModel

import torch
import torch.nn.functional as F
from torch import nn
from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encodingW


class RoBerta(nn.Module):
    def __init__(self, name: str, train_bert: bool, enc_num):
        super().__init__()
        if name == "roberta-base":
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        self.roberta = RobertaModel.from_pretrained(name)
        if not train_bert:
            for parameter in self.roberta.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        encoded_text = self.roberta(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
        # use the output of the X-th transformer encoder layers
        xs = encoded_text.last_hidden_state

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask

        return xs, mask

def build_roberta(args):
    # position_embedding = build_position_encoding(args)
    train_roberta = args.lr_bert > 0
    roberta = RoBerta(args.roberta_model, train_roberta, args.bert_enc_num)
    return roberta
