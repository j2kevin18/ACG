from functools import partial
from timm.models import xception
from model.common import SeparableConv2d, Block
from model.OT import OTBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

encoder_params = {
    "xception": {
        "features": 2048,
        "init_op": partial(xception, pretrained=True)
    }
}


class ACG(nn.Module):
    """ End-to-End Reconstruction-Classification Learning for Face Forgery Detection """

    def __init__(self, num_classes, drop_rate=0.2):
        super(ACG, self).__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = encoder_params[self.name]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(encoder_params[self.name]["features"], num_classes)
        self.encoder_ae1 = Block(728, 182, 3, 1)
        self.encoder_ae2 = Block(182, 32, 3, 1)
        self.encoder_ae3 = Block(32, 8, 3, 1)
        self.ot = OTBlock()
        self.decoder_ae1 = Block(8, 32, 3, 1)
        self.decoder_ae2 = Block(32, 182, 3, 1)
        self.decoder_ae3 = Block(182, 728, 3, 1)
        self.encoder.fc.requires_grad_(False)


        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def forward(self, x):
        # clear the loss inputs
        self.loss_inputs = dict(recons=[], contra=[])
        noise_x = self.add_white_noise(x) if self.training else x
        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        out = self.encoder.block1(out)
        out = self.encoder.block2(out)
        out = self.encoder.block3(out)
        out = self.encoder.block4(out)
        embedding = self.encoder_ae1(out)
        embedding = self.encoder_ae2(embedding)
        embedding = self.encoder_ae3(embedding)
        embedding = self.ot(embedding)
        
        embedding = self.decoder_ae1(embedding)
        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)
        
        embedding = self.decoder_ae2(embedding)
        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)
        
        embedding = self.decoder_ae3(embedding)
        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)

        out = self.dropout(embedding)
        out = self.decoder1(out)
        out_d2 = self.decoder2(out)

        norm_embed, corr = self.norm_n_corr(out_d2)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder3(out_d2)
        out_d4 = self.decoder4(out)
        

        norm_embed, corr = self.norm_n_corr(out_d4)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder5(out_d4)
        pred = self.decoder6(out)

        recons_x = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_inputs['recons'].append(recons_x)

        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)

        embedding = self.encoder.block8(embedding)
        embedding = self.encoder.block9(embedding)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        embedding = self.global_pool(embedding).squeeze()

        out = self.dropout(embedding)
        return self.fc(out)
