"""Alternative gesture encoders compatible with the frozen HMOG/FETA cache.

Inputs: batch dict with touch [B,T,6], accel/gyro [B,T,5], and *_mask.
Outputs: L2-normalized [B,embedding_dim]. All temporal reductions mask padding.

The `baseline` model delegates to the exact original `gesture_model.GestureEncoder`.
The original baseline ALREADY has three dilated residual convolution blocks (1,2,4).
The `tcn` variant is a deeper, two-convolution-per-block dilated TCN with (1,2,4,8,16).
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from gesture_model import GestureEncoder as BaselineGestureEncoder
from gesture_model import ResidualTimeBlock

ARCHITECTURES=('baseline','tcn','attention_pool','crossmodal')


def _moments(x,mask,weight=None):
    m=mask.to(dtype=x.dtype).unsqueeze(-1)
    if weight is None:
        weight=m/m.sum(dim=1,keepdim=True).clamp_min(1.)
    else:
        weight=weight.unsqueeze(-1)*m
        weight=weight/weight.sum(dim=1,keepdim=True).clamp_min(1.e-8)
    mean=(weight*x).sum(1)
    var=(weight*(x-mean[:,None,:]).square()).sum(1)
    return torch.cat((mean,torch.sqrt(var+1.e-6)),dim=-1)


class DeepResidualTCNUnit(nn.Module):
    def __init__(self,width,dilation,dropout=0.08):
        super().__init__()
        self.conv1=nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.norm1=nn.LayerNorm(width)
        self.conv2=nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.norm2=nn.LayerNorm(width)
        self.drop=nn.Dropout(dropout)

    def forward(self,x,mask):
        m=mask.unsqueeze(-1)
        y=self.conv1(x.transpose(1,2)).transpose(1,2)
        y=self.drop(F.gelu(self.norm1(y)))*m
        y=self.conv2(y.transpose(1,2)).transpose(1,2)
        y=self.drop(F.gelu(self.norm2(y)))
        return (x+y)*m


class Branch(nn.Module):
    def __init__(self,features,width,mode):
        super().__init__()
        self.project=nn.Linear(features,width)
        if mode=='tcn':
            self.blocks=nn.ModuleList([DeepResidualTCNUnit(width,d) for d in (1,2,4,8,16)])
        else:
            self.blocks=nn.ModuleList([ResidualTimeBlock(width,d) for d in (1,2,4)])
        self.attention=nn.Linear(width,1) if mode=='attention_pool' else None
        self.reduce=nn.Linear(width*2,width)

    def tokens(self,x,mask):
        mask=mask.bool()
        x=F.gelu(self.project(x))*mask.unsqueeze(-1)
        for layer in self.blocks:
            x=layer(x,mask)
        return x

    def forward(self,x,mask):
        x=self.tokens(x,mask)
        if self.attention is not None:
            # Every cache gesture has >=3 valid samples, so rows are never fully masked.
            w=torch.softmax(self.attention(x).squeeze(-1).masked_fill(~mask,-1.e4),dim=1)
            pooled=_moments(x,mask,w)
        else:
            pooled=_moments(x,mask)
        return F.gelu(self.reduce(pooled))


def _resample_on_grid(x,t,mask,query_t):
    """Differentiable (w.r.t. x) linear interpolation on the real elapsed-time axis.

    No attempt is made to align sequences by sample INDEX. Query times are the
    touch-defined gesture interval. Sensor endpoints are held when the sensor
    coverage ends early. Inputs have sorted nonnegative elapsed_s (dataset invariant).
    """
    if not bool(mask[:,0].all()):
        raise ValueError('Every modality needs at least one unmasked row')
    t=t.float()
    if torch.any((t[:,1:]-t[:,:-1])[(mask[:,1:] & mask[:,:-1])]<-1e-5):
        raise ValueError('Elapsed time must be nondecreasing')
    t_valid=t.masked_fill(~mask,float('inf')).contiguous()
    high=torch.searchsorted(t_valid,query_t.contiguous(),right=False)
    last=mask.sum(dim=1).long().sub(1)[:,None]
    high=torch.minimum(high,last).clamp(min=0)
    low=(high-1).clamp(min=0)
    n,steps,channels=x.shape
    take_h=high.unsqueeze(-1).expand(-1,-1,channels)
    take_l=low.unsqueeze(-1).expand(-1,-1,channels)
    a=x.gather(1,take_l);b=x.gather(1,take_h)
    t0=t.gather(1,low);t1=t.gather(1,high)
    alpha=((query_t-t0)/(t1-t0).clamp_min(1e-6)).clamp(0,1).unsqueeze(-1)
    return a+(b-a)*alpha


class CrossModalEncoder(nn.Module):
    def __init__(self,width,embedding_dim,grid_points=32,heads=4):
        super().__init__()
        if width%heads:raise ValueError('width must be divisible by heads')
        if grid_points<4:raise ValueError('grid_points must be >=4')
        self.grid_points=grid_points
        self.touch=Branch(6,width,'baseline')
        self.accel=Branch(5,width,'baseline')
        self.gyro=Branch(5,width,'baseline')
        self.to_acc=nn.MultiheadAttention(width,heads,dropout=0.05,batch_first=True)
        self.to_gyro=nn.MultiheadAttention(width,heads,dropout=0.05,batch_first=True)
        self.cross_norm=nn.LayerNorm(width)
        self.fusion=nn.Sequential(nn.Linear(width*4,192),nn.GELU(),nn.Dropout(.10),
                                  nn.Linear(192,embedding_dim))

    def forward(self,batch):
        vectors={}
        for kind,branch in [('touch',self.touch),('accel',self.accel),('gyro',self.gyro)]:
            x=batch[kind]; mask=batch[kind+'_mask'].bool()
            tokens=branch.tokens(x,mask)
            # Raw elapsed_s is untouched by the train-fitted sensor normalization.
            elapsed=x[...,4] if kind=='touch' else x[...,3]
            vectors[kind]=(tokens,elapsed,mask)
        touch_t=vectors['touch'][1]
        touch_mask=vectors['touch'][2]
        duration=touch_t.masked_fill(~touch_mask,0).amax(dim=1,keepdim=True)
        grid=duration*torch.linspace(0,1,self.grid_points,device=touch_t.device,dtype=touch_t.dtype)[None,:]
        aligned={kind:_resample_on_grid(*vectors[kind],grid) for kind in ('touch','accel','gyro')}
        index=torch.arange(self.grid_points,device=touch_t.device)
        # Local cross attention: no interactions between distant timestamps.
        exclude=(index[:,None]-index[None,:]).abs()>2
        ta,_=self.to_acc(aligned['touch'],aligned['accel'],aligned['accel'],attn_mask=exclude,need_weights=False)
        tg,_=self.to_gyro(aligned['touch'],aligned['gyro'],aligned['gyro'],attn_mask=exclude,need_weights=False)
        cross=self.cross_norm(aligned['touch']+ta+tg)
        feat=torch.cat((aligned['touch'].mean(1),aligned['accel'].mean(1),
                        aligned['gyro'].mean(1),cross.mean(1)),dim=1)
        return F.normalize(self.fusion(feat),dim=1,eps=1e-8)


class ThreeBranchEncoder(nn.Module):
    def __init__(self,width,embedding_dim,mode):
        super().__init__()
        self.touch=Branch(6,width,mode)
        self.accel=Branch(5,width,mode)
        self.gyro=Branch(5,width,mode)
        self.fusion=nn.Sequential(nn.Linear(width*3,192),nn.GELU(),nn.Dropout(.10),
                                  nn.Linear(192,embedding_dim))

    def forward(self,batch):
        z=torch.cat([self.touch(batch['touch'],batch['touch_mask']),
                     self.accel(batch['accel'],batch['accel_mask']),
                     self.gyro(batch['gyro'],batch['gyro_mask'])],dim=1)
        return F.normalize(self.fusion(z),dim=1,eps=1e-8)


def make_encoder(architecture:dict):
    """Instantiate from the checkpoint's architecture field, old or new."""
    name=architecture.get('name','baseline')
    if name not in ARCHITECTURES:raise ValueError(f'Unknown architecture: {name}')
    width=int(architecture.get('width',48))
    dim=int(architecture.get('embedding_dim',64))
    if name=='baseline':return BaselineGestureEncoder(width,dim)
    if name=='crossmodal':
        return CrossModalEncoder(width,dim,int(architecture.get('grid_points',32)),
                                 int(architecture.get('attention_heads',4)))
    return ThreeBranchEncoder(width,dim,mode=('tcn' if name=='tcn' else 'attention_pool'))


def prototypical_cross_session_loss(embedding,labels,metadata,temperature=0.1):
    """Optional episodic training: support and query from distinct TRAIN sessions.

    Classes are P distinct training identities from the PK sampler. For each
    identity one support session and one different query session are chosen;
    all queries classify against the P session-specific prototypes.
    """
    if temperature<=0:raise ValueError('Temperature must be positive')
    all_labels=torch.unique(labels,sorted=True)
    if len(all_labels)<2:raise ValueError('Need >=2 identities')
    prototypes=[];queries=[];targets=[]
    for class_index,label in enumerate(all_labels):
        rows=torch.where(labels==label)[0].tolist()
        sessions=sorted({str(metadata[i]['session_id']) for i in rows})
        if len(sessions)<2:raise ValueError('Each PK identity needs >=2 training sessions')
        # Deterministically assign the two sessions; sampler changes participants
        # and sampled sessions every training step/epoch.
        support=[i for i in rows if str(metadata[i]['session_id'])==sessions[0]]
        query=[i for i in rows if str(metadata[i]['session_id'])==sessions[1]]
        proto=F.normalize(embedding[support].mean(dim=0,keepdim=True),dim=1)
        prototypes.append(proto)
        queries.append(embedding[query])
        targets.extend([class_index]*len(query))
    proto=torch.cat(prototypes,0)
    q=torch.cat(queries,0)
    logits=q.float()@proto.float().T/temperature
    return F.cross_entropy(logits,torch.tensor(targets,device=labels.device))
