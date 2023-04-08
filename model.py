import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
import copy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

class Modality_Enhancement_Module(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim = 1024
        self.channel_conv1 = nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
    def forward(self,vfeat,ffeat):
        channel_attn = self.channel_conv1(vfeat)
        bit_wise_attn = self.channel_conv1(ffeat)
        
        filter_feat = torch.sigmoid(channel_attn)*torch.sigmoid(bit_wise_attn)*vfeat
        
        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat


class Optical_convolution(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim = 1024
        self.opt_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
    def forward(self,ffeat):
        opt_wise_attn = self.opt_wise_attn(ffeat)
        filter_ffeat = torch.sigmoid(opt_wise_attn)*ffeat
        opt_attn = self.attention(filter_ffeat)
        return opt_attn, filter_ffeat


class TFE_DC_Module(nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        
        embed_dim = 1024
        self.layer1 = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=2 ** 0, dilation=2 ** 0),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        self.layer2 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 1, dilation=2 ** 1),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        self.layer3 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 2, dilation=2 ** 2),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
    def forward(self, x):
        out = self.layer1(x)
        out_attention1 = self.attention(torch.sigmoid(out)*x)
        
        out = self.layer2(out)
        out_attention2 = self.attention(torch.sigmoid(out)*x)
        
        out = self.layer3(out)
        out_feature = torch.sigmoid(out)*x
        out_attention3 = self.attention(out_feature)
        
        out_attention = (out_attention1+out_attention2+out_attention3)/3.0
        
        
        return out_attention, out_feature


class TFEDCN(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        mid_dim=1024
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio

        self.vAttn = getattr(model,args['opt'].AWM)(1024,args)
        self.fAttn = getattr(model,args['opt'].TCN)(1024,args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class+1, 1))
        
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
        self.batch_avg=nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        
        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n=feat.size()
        f_atn,ffeat = self.fAttn(feat[:,1024:,:])
        v_atn,vfeat = self.vAttn(feat[:,:1024,:],ffeat)
        x_atn = (f_atn+v_atn)/2
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls.transpose(-1, -2), 'attn':x_atn.transpose(-1, -2), 'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2)}


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn= outputs['feat'],outputs['cas'],outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        
        
        
        mutual_loss=0.5*F.mse_loss(v_atn,f_atn.detach())+0.5*F.mse_loss(f_atn,v_atn.detach())
        b,n,c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, f_atn,include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)

        v_loss_norm = v_atn.abs().mean()
        # guide loss
        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.abs().mean()
        # guide loss
        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()
        
        
        # total loss
        total_loss = (loss_mil_orig.mean() + 
                      loss_mil_supp.mean()
                      +args['opt'].alpha1*(f_loss_norm+v_loss_norm)
                      +args['opt'].alpha2*f_loss_guide
                      +args['opt'].alpha3*v_loss_guide
                      +args['opt'].alpha4*mutual_loss
                      )

        return total_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind