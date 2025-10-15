import torch
import numpy as np
import torch.nn.functional as F
import timm
from torch import nn


def load_pretrained_model(model_path, out_dim, cfg, device='cuda'):
    # I changed how the configs are saved so for now we need to check for old versions

    if cfg['exp_details']['dataset'] == 'ADNI': model = MRIEncoder(in_channel=1, feat_dim=1024, out_dim=out_dim, return_emb=True, device=device)
    else: 
        model = Network(out_dim, in_dim=cfg['model']['in_dim'], hid_dim=cfg['model']['mlp_hid_dim'], backbone=cfg['model']['backbone'])
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model

class Network(nn.Module):
    def __init__(self, out_features, in_dim=-1, hid_dim=32, return_emb=True, backbone='resnet18'):
        super().__init__()

        if backbone == 'resnet18':
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            self.backbone.fc = nn.Identity(512)
            self.head = nn.Linear(512, out_features)  
        elif backbone == 'resnet50':
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            self.backbone.fc = nn.Identity(2048)
            self.head = nn.Linear(2048, out_features)  
        elif backbone == 'vit':
            self.backbone = timm.create_model('vit_base_patch16_224', checkpoint_path='/users/jdt0025/timm_models/vit.pt')
            self.backbone.head = nn.Identity(768)
            self.head = nn.Linear(768, out_features)
        elif backbone == 'mlp':
            if in_dim < 0: raise ValueError(f'Parameter in_dim should be greater than 0 when backbone=mlp')
            self.backbone = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(hid_dim),
                                          nn.Dropout(0.1),
                                          nn.Linear(hid_dim, hid_dim),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(hid_dim),
                                          nn.Dropout(0.1))
            self.head = nn.Linear(hid_dim, out_features)
        else:
            raise NotImplementedError(f'Backbone model {backbone} not implmented! Use one of [resnet18, vit, resnet50]')

        self.return_emb = return_emb


    def forward(self, x):
        emb = self.backbone(x)
        out = self.head(F.relu(emb))

        if self.return_emb: return out, emb
        else: return out



class MRIEncoder(nn.Module):

    def __init__(self, 
                 in_channel =1, 
                 feat_dim   = 1024,
                 out_dim    = 10,
                 expansion  = 4,
                 dropout    = 0.5,  
                 norm_type  = 'Instance', 
                 activation = 'relu',
                 return_emb = True,
                 device     = 'cuda'):
        super(MRIEncoder, self).__init__()
    
        assert activation in ['relu', 'selu'], f'Expected param "activation" to be in [relu, selu], got {activation}'

        self.device = device
        self.feat_dim = feat_dim
        self.return_emb = return_emb

        if activation == 'relu': activation_fn = nn.ReLU(inplace=True)
        else: activation_fn = nn.SELU(inplace=True)

        self.conv = nn.Sequential()

        # BLOCK 1
        self.conv.add_module('conv0_s1',nn.Conv3d(in_channel, 4*expansion, kernel_size=1))

        if norm_type == 'Instance':
           self.conv.add_module('lrn0_s1',nn.InstanceNorm3d(4*expansion))
        else:
           self.conv.add_module('lrn0_s1',nn.BatchNorm3d(4*expansion))

        self.conv.add_module('relu0_s1', activation_fn)
        self.conv.add_module('pool0_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('dropout_1', nn.Dropout(dropout))


        # BLOCK 2
        self.conv.add_module('conv1_s1',nn.Conv3d(4*expansion, 32*expansion, kernel_size=3, padding=0, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn1_s1',nn.InstanceNorm3d(32*expansion))
        else:
            self.conv.add_module('lrn1_s1',nn.BatchNorm3d(32*expansion))
            
        self.conv.add_module('relu1_s1', activation_fn)
        self.conv.add_module('pool1_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('dropout_2', nn.Dropout(dropout))

        # BLOCK 3
        self.conv.add_module('conv2_s1',nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn2_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu2_s1', activation_fn)
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('dropout_3', nn.Dropout(dropout))

        # BLOCK 4
        self.conv.add_module('conv3_s1',nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn3_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu3_s1', activation_fn)
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))

        self.conv.add_module('dropout_4', nn.Dropout(dropout))
        self.conv.add_module('Flatten', nn.Flatten())
        self.conv.add_module('head_s1',nn.Linear(64*expansion*5*5*5, feat_dim))

        # PROJECTION
        self.head = nn.Sequential()
        self.head.add_module('head_relu', nn.ReLU(inplace=True))
        self.head.add_module('head_s2', nn.Linear(feat_dim, 512))
        self.head.add_module('head_relu', nn.ReLU(inplace=True))
        self.head.add_module('head_s3', nn.Linear(512, out_dim))
        

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)['state_dict']
        pretrained_dict = {k[6:]: v for k, v in list(pretrained_dict.items()) if k[6:] in model_dict and 'conv3_s1' not in k and 'fc6' not in k and 'fc7' not in k and 'fc8' not in k}

        model_dict.update(pretrained_dict)
        

        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])
        return pretrained_dict.keys()

    def freeze(self, pretrained_dict_keys):
        for name, param in self.named_parameters():
            if name in pretrained_dict_keys:
                param.requires_grad = False
                

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x):
        emb = self.conv(x)
        out = self.head(emb)
        
        if self.return_emb: return out, emb
        
        return out

def weights_init(model):
    if type(model) in [nn.Conv3d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)