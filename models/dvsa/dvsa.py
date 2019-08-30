import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from functools import partial
import os

from models.dvsa.dvsa_utils.transformer import Transformer

class DVSA(nn.Module):

#    def __init__(self, num_class, input_size=2048, enc_size=128, dropout=0.2, hidden_size=256, n_layers=1, n_heads=4, attn_drop=0.2, num_frm=5, has_loss_weighting=False):
    def __init__(self, **kwargs):
        super().__init__()
        num_class          = kwargs['num_class']
        input_size         = kwargs['input_size']
        enc_size           = kwargs['enc_size']
        dropout            = kwargs['dropout']
        hidden_size        = kwargs['hidden_size']
        n_layers           = kwargs['n_layers']
        n_heads            = kwargs['n_heads']
        attn_drop          = kwargs['attn_drop']
        num_frm            = kwargs['num_frm']
        has_loss_weighting = kwargs['has_loss_weighting']
        
        # encode the region feature
        self.feat_enc = nn.Sequential(
            nn.Linear(input_size, enc_size),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.sigmoid = nn.Sigmoid()

        # lookup table for object label embedding
        self.obj_emb = nn.Embedding(num_class+1, enc_size) # +1 for the dummy paddings
        self.num_class = num_class

        self.obj_interact = Transformer(enc_size, 0, 0,
            d_hidden=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            drop_ratio=attn_drop)

        self.obj_interact_fc = nn.Sequential(
            nn.Linear(enc_size*2, int(enc_size/2)),
            nn.ReLU(),
            nn.Linear(int(enc_size/2), 5), # object interaction guidance (always 5 snippets)
            nn.Sigmoid()
        )

        self.num_frm = num_frm 
        self.has_loss_weighting = has_loss_weighting

        if isinstance(kwargs['pretrained'], int) and kwargs['pretrained']:
            self._load_pretrained_weights()

    def forward(self, x_o, obj, load_type):
        is_evaluate = 1 if load_type[0] == 'test' or load_type[0] == 'val' else 0
        if is_evaluate:
            return self.output_attn(x_o, obj)

        x_o = self.feat_enc(x_o.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

        x_o = torch.stack([x_o[0], x_o[1], x_o[0]])
        obj = torch.stack([obj[0], obj[0], obj[1]])

        N, C_out, T, num_proposals = x_o.size()
        assert(N == 3) # two pos samples and one neg sample

        # attention
        O = obj.size(1)
        attn_key = self.obj_emb(obj)

        num_pos_obj = torch.sum(obj[0]<self.num_class).long().item()
        num_neg_obj = torch.sum(obj[2]<self.num_class).long().item()
        # object interaction guidance
        attn_key_frm_feat = attn_key[0:1, :num_pos_obj] # cat visual feature
        obj_attn_emb,_ = self.obj_interact(attn_key_frm_feat)
        obj_attn_emb = obj_attn_emb[:, :num_pos_obj, :]
        obj_attn_emb = torch.cat((obj_attn_emb, attn_key[0:1, :num_pos_obj], ), dim=2)
        obj_attn_emb = self.obj_interact_fc(obj_attn_emb) # N, O, 5
        
        itv = math.ceil(T/5)
        tmp = [] # expand obj_attn_emb to N, O, T
        for i in range(5):
            l = min(itv*(i+1), T)-itv*i
            if l>0:
                tmp.append(obj_attn_emb[:, :, i:(i+1)].expand(1, num_pos_obj, l))
        obj_attn_emb = torch.cat(tmp, 2).squeeze(0)
        assert(obj_attn_emb.size(1) == self.num_frm)

        loss_weigh = torch.mean(obj_attn_emb, dim=0)
        loss_weigh = torch.cat((loss_weigh, loss_weigh)).unsqueeze(1)

        if self.has_loss_weighting:
            # dot-product attention
            x_o = x_o.view(N, 1, C_out, T, num_proposals)
            attn_weights = self.sigmoid((x_o*attn_key.view(N, O, C_out, 1, 1)).sum(2)/math.sqrt(C_out))

            pos_weights = attn_weights[0, :num_pos_obj, :, :]
            neg1_weights = attn_weights[1, :num_pos_obj, :, :]
            neg2_weights = attn_weights[2, :num_neg_obj, :, :]

            return torch.cat((torch.stack((torch.mean(torch.max(pos_weights, dim=2)[0], dim=0), torch.mean(torch.max(neg1_weights, dim=2)[0], dim=0)), dim=1),
                torch.stack((torch.mean(torch.max(pos_weights, dim=2)[0], dim=0), torch.mean(torch.max(neg2_weights, dim=2)[0], dim=0)), dim=1))), loss_weigh
        else:
            # dot-product attention
            x_o = x_o.view(N, 1, C_out, T*num_proposals)
            attn_weights = self.sigmoid((x_o*attn_key.view(N, O, C_out, 1)).sum(2)/math.sqrt(C_out))

            pos_weights = attn_weights[0, :num_pos_obj, :]
            neg1_weights = attn_weights[1, :num_pos_obj, :]
            neg2_weights = attn_weights[2, :num_neg_obj, :]

            return torch.stack((torch.stack((torch.mean(torch.max(pos_weights, dim=1)[0]), torch.mean(torch.max(neg1_weights, dim=1)[0]))),
                torch.stack((torch.mean(torch.max(pos_weights, dim=1)[0]), torch.mean(torch.max(neg2_weights, dim=1)[0]))))), loss_weigh

    def output_attn(self, x_o, obj):
        x_o = self.feat_enc(x_o.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

        N, C_out, T, num_proposals = x_o.size()
        assert(N == 1) # two pos samples and one neg sample

        # attention
        O = obj.size(1)
        attn_key = self.obj_emb(obj)

        # dot-product attention
        x_o = x_o.view(N, 1, C_out, T*num_proposals)
        attn_weights = self.sigmoid((x_o*attn_key.view(N, O, C_out, 1)).sum(2)/math.sqrt(C_out))
        # attn_weights = self.sigmoid((x_e*attn_key.view(N, O, C_out, 1).expand(N, O, C_out, T*num_proposals)).sum(2)) # N, O, T, H*W

        # additive attention
        # x_e = x_o.view(N, 1, C_out, T, H*W).contiguous().expand(N, O, C_out, T, H*W)
        # attn_e = attn_key.view(N, O, C_out, 1, 1).expand(N, O, C_out, T, H*W)
        # attn_weights = self.attn_mlp(torch.cat((x_e, attn_e), dim=2).permute(0,1,3,4,2).contiguous()).squeeze(4) # N, O, T, H*W

        return attn_weights.view(N, O, T, num_proposals)

    def _load_pretrained_weights(self):
        state_dict = torch.load('weights/yc2bb_full-model.pth', map_location=lambda storage, location: storage)

        self.load_state_dict(state_dict)

