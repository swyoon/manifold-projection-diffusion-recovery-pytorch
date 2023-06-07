import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode, ToTensor
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class LatentAE(nn.Module):
    """An autoencoder on the representation space of a backbone"""
    def __init__(self, ae, backbone_name='vit_base_patch16_224', spherical=True, centercrop=False,
                 warmup_iter=1000, i_iter=0):
        """
        spherical: project the representation to the unit sphere
        centercrop: use centercrop in preprocessing
        """
        super().__init__()
        self.ae = ae
        self.backbone_name = backbone_name
        assert backbone_name in {'vit_base_patch16_224', 'resnetv2_50x1_bitm'}
        self.spherical = spherical
        self.centercrop = centercrop

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        self.warmup_iter = warmup_iter
        self.i_iter = i_iter

    def get_transform(self):
        config = resolve_data_config({}, model=self.backbone)
        transform = create_transform(**config)
        if self.centercrop:
            return transform
        else:
            if self.backbone_name == 'vit_base_patch16_224':
                return Compose([Resize(224, interpolation=InterpolationMode.BICUBIC),
                                ToTensor(),
                                Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
            else:
                raise NotImplementedError

    def forward(self, x):
        """anomaly score"""
        z = self.backbone(x)
        z = self._project(z)
        return self.ae(z)

    def predict(self, x):
        return self(x)

    def encode(self, x):
        z = self.backbone(x)
        return self._project(z)

    def _project(self, x):
        if self.spherical:
            return x / x.norm(dim=1, keepdim=True)
        return x

    def train_step(self, x, opt, fix_backbone=False, **kwargs):
        self.train()
        opt.zero_grad()
        if fix_backbone:
            with torch.no_grad():
                self.backbone.eval()
                z = self.backbone(x)
        else:
            z = self.backbone(x)
        z = self._project(z)
        zz = self.ae.encode(z)
        r = self.ae.decode(zz)
        recon_error = ((r - z) ** 2).sum(dim=1).mean()  # use sum 
        recon_error.backward()
        opt.step()
        d_train = {'loss': recon_error.item()}
        return d_train 
