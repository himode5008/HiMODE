from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from module.attention import MultiheadAttentionRelative
from Utils.misc import get_clones

layer_idx = 0

class Transformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 6):
        super().__init__()

        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, nhead)
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim, nhead)
        self.cross_attn_layers = get_clones(cross_attn_layer, num_attn_layers)

        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

    def _alternating_attn(self, feat: torch.Tensor, pos_enc: torch.Tensor, pos_indexes: Tensor, hn: int):

        global layer_idx

        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            layer_idx = idx

            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn

            feat = checkpoint(create_custom_self_attn(self_attn), feat, pos_enc, pos_indexes)


            if idx == self.num_attn_layers - 1:

                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, True)

                    return custom_cross_attn
            else:

                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, False)

                    return custom_cross_attn

            feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:], pos_enc,
                                           pos_indexes)

        layer_idx = 0
        return attn_weight

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor, pos_enc: Optional[Tensor] = None):


        bs, c, hn, w = feat_left.shape

        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        if pos_enc is not None:
            with torch.no_grad():

                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()
        else:
            pos_indexes = None

        feat = torch.cat([feat_left, feat_right], dim=1)

        attn_weight = self._alternating_attn(feat, pos_enc, pos_indexes, hn)
        attn_weight = attn_weight.view(hn, bs, w, w).permute(1, 0, 2, 3)

        return attn_weight


class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.self_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, feat: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None):

        feat2 = self.norm1(feat)

        feat2, attn_weight, _ = self.self_attn(query=feat2, key=feat2, value=feat2, pos_enc=pos,
                                               pos_indexes=pos_indexes)

        feat = feat + feat2

        return feat


class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, feat_left: Tensor, feat_right: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None,
                last_layer: Optional[bool] = False):

        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)

        if pos is not None:
            pos_flipped = torch.flip(pos, [0])
        else:
            pos_flipped = pos
        feat_right_2 = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped,
                                       pos_indexes=pos_indexes)[0]

        feat_right = feat_right + feat_right_2


        if last_layer:
            w = feat_left_2.size(0)
            attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)
        else:
            attn_mask = None

        feat_right_2 = self.norm2(feat_right)
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2,
                                                             attn_mask=attn_mask, pos_enc=pos,
                                                             pos_indexes=pos_indexes)


        feat_left = feat_left + feat_left_2


        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        return feat, raw_attn

    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):

        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask


def build_transformer(args):
    return Transformer(
        hidden_dim=args.channel_dim,
        nhead=args.nheads,
        num_attn_layers=args.num_attn_layers
    )
