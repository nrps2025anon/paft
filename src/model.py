"""Backbone, optional narrow penultimate layer, main head and auxiliary branch.

The auxiliary branch is a linear bottleneck of width d followed by a classification head.
The narrow-penultimate variant instead replaces the penultimate layer itself by a linear
map to the target width, which bounds its rank architecturally rather than by a penalty;
it is linear on purpose, since a nonlinearity would blur that bound.

Optional modules are built only when they are requested, so cells that do not use them
consume the random number stream in exactly the same order.
"""
import torch
import torch.nn as nn
import timm


class SweepNet(nn.Module):
    def __init__(self, backbone_id, num_classes, aux, proj_dim=2, neck_dim=None,
                 centers=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_id, pretrained=True, num_classes=0)
        d = self.backbone.num_features
        self.neck = None
        if neck_dim:
            self.neck = nn.Linear(d, int(neck_dim))
            d = int(neck_dim)
        self.feat_dim = d
        self.main_head = nn.Linear(d, num_classes)
        self.aux = aux
        if aux:
            self.proj_2d = nn.Linear(d, proj_dim)
            self.aux_head = nn.Linear(proj_dim, num_classes)
        if centers:
            self.register_buffer('centers', torch.randn(num_classes, d))

    def forward(self, x):
        f = self.backbone(x)
        if self.neck is not None:
            f = self.neck(f)
        m = self.main_head(f)
        if not self.aux:
            return m, None, f, None
        p = self.proj_2d(f)
        return m, self.aux_head(p), f, p
