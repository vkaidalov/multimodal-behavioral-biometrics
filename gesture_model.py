"""Compact 3-branch, variable-length temporal gesture encoder; 64-D embedding."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ResidualTimeBlock(nn.Module):
    def __init__(self,channels,dilation,dropout=0.08):
        super().__init__()
        self.conv=nn.Conv1d(channels,channels,5,padding=2*dilation,dilation=dilation)
        self.norm=nn.LayerNorm(channels)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask):
        # LayerNorm per timestamp, not batch norm across padded positions.
        y=self.conv(x.transpose(1,2)).transpose(1,2)
        y=self.dropout(F.gelu(self.norm(y)))
        return (x+y)*mask.unsqueeze(-1)


class TimeBranch(nn.Module):
    def __init__(self,features,width):
        super().__init__()
        self.project=nn.Linear(features,width)
        self.blocks=nn.ModuleList([ResidualTimeBlock(width,d) for d in (1,2,4)])
        self.out=nn.Linear(width*2,width)

    def forward(self,x,mask):
        mask=mask.bool()
        x=F.gelu(self.project(x))*mask.unsqueeze(-1)
        for blk in self.blocks:x=blk(x,mask)
        n=mask.sum(dim=1,keepdim=True).clamp_min(1)
        mean=x.sum(dim=1)/n
        var=((x-mean.unsqueeze(1)).square()*mask.unsqueeze(-1)).sum(dim=1)/n
        return F.gelu(self.out(torch.cat([mean,torch.sqrt(var+1e-6)],dim=-1)))


class GestureEncoder(nn.Module):
    def __init__(self,width=48,embedding_dim=64):
        super().__init__()
        self.width=width;self.embedding_dim=embedding_dim
        self.touch=TimeBranch(6,width)
        self.accel=TimeBranch(5,width)
        self.gyro=TimeBranch(5,width)
        self.fusion=nn.Sequential(nn.Linear(width*3,192),nn.GELU(),nn.Dropout(0.10),
                                  nn.Linear(192,embedding_dim))

    def forward(self,batch):
        z=torch.cat([self.touch(batch['touch'],batch['touch_mask']),
                     self.accel(batch['accel'],batch['accel_mask']),
                     self.gyro(batch['gyro'],batch['gyro_mask'])],dim=1)
        return F.normalize(self.fusion(z),dim=1,eps=1e-8)


def supervised_contrastive_loss(embedding,labels,temperature=0.1):
    # Numerically stable logsumexp; positives cannot include the anchor itself.
    logits=embedding.float()@embedding.float().T/temperature
    b=len(labels)
    eye=torch.eye(b,dtype=torch.bool,device=labels.device)
    positive=(labels[:,None]==labels[None,:]) & ~eye
    if not torch.all(positive.sum(dim=1)>0):raise ValueError('Missing positives in batch')
    logits=logits.masked_fill(eye,float('-inf'))
    log_prob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    return -(log_prob.masked_fill(~positive,0).sum(dim=1)/positive.sum(dim=1)).mean()
