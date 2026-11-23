"""Representation-specific supervision and split-isolated, confidence-gated distillation."""
import torch
from torch.nn import functional as F


def masked_mean(values, valid):
    return (values*valid).sum()/valid.expand_as(values).sum().clamp_min(1)


def robust(a,b,beta=2.):
    return F.smooth_l1_loss(a,b,reduction='none',beta=beta)


def supervised(output,y,valid,settings=None):
    s=settings or {};pred=output['y'].float()
    weights=torch.tensor(s.get('line_weights',[1.,1.]),device=y.device)[None,:,None]*valid
    coord=masked_mean(robust(pred,y,s.get('coord_beta',2.)),weights)
    adjacent=valid[:,:,1:]&valid[:,:,:-1]
    slope=masked_mean(robust(pred.diff(dim=-1),y.diff(dim=-1),1.),adjacent)
    both=valid.all(1)
    thickness=masked_mean(robust(pred[:,1]-pred[:,0],y[:,1]-y[:,0]),both)
    loss=coord+s.get('slope_weight',.2)*slope+s.get('thickness_weight',.15)*thickness
    method=output['method']
    if method=='signed_distance':
        field=output['field'];h,w=field.shape[-2:]
        target_y=F.interpolate(y,size=w,mode='linear',align_corners=False)/255
        mask=(F.interpolate(valid.float(),size=w,mode='nearest')>.5)[:,:,None,:].expand_as(field)
        grid=(torch.arange(h,device=y.device).float()+.5)/h
        target=grid[None,None,:,None]-target_y[:,:,None,:]
        near=1+3*torch.exp(-target.abs()/.08)
        distance=masked_mean(robust(field,target,.03)*16,mask*near)
        vertical=masked_mean(robust(field.diff(dim=2)*h,torch.ones_like(field[:,:,1:]),.5),mask[:,:,1:]&mask[:,:,:-1])
        loss=loss+s.get('field_weight',1.)*distance+s.get('vertical_weight',.05)*vertical
    elif method=='spline_queries':
        triple=valid[:,:,2:]&valid[:,:,1:-1]&valid[:,:,:-2]
        curvature=masked_mean(robust(pred.diff(n=2,dim=-1),y.diff(n=2,dim=-1),.5),triple)
        k=output['controls'].shape[-1]
        low_pred=F.adaptive_avg_pool1d(pred, k)
        low_gt=F.adaptive_avg_pool1d(y*valid,k)/F.adaptive_avg_pool1d(valid.float(),k).clamp_min(1e-6)
        low_valid=F.adaptive_avg_pool1d(valid.float(),k)>.5
        loss=loss+s.get('curvature_weight',.3)*curvature+s.get('coarse_weight',.25)*masked_mean(robust(low_pred,low_gt),low_valid)
    elif method=='iterative_refine':
        earlier=output['trajectory'][:-1]
        total=sum(range(1,len(earlier)+1))
        intermediate=sum((i+1)*masked_mean(robust(p,y),weights) for i,p in enumerate(earlier))/max(total,1)
        loss=loss+s.get('intermediate_weight',.4)*intermediate
    elif method=='axial_distribution':
        logits=output['distribution_logits'];bins=logits.shape[2]
        pos=(y/255*(bins-1)).clamp(0,bins-1);lo=pos.floor().long();hi=(lo+1).clamp_max(bins-1)
        target=torch.zeros_like(logits)
        target.scatter_add_(2,lo[:,:,None,:],(1-(pos-lo))[:,:,None,:])
        target.scatter_add_(2,hi[:,:,None,:],(pos-lo)[:,:,None,:])
        ce=-(target*logits.log_softmax(2)).sum(2)
        emd=(logits.softmax(2).cumsum(2)-target.cumsum(2)).abs().mean(2)*255
        loss=loss+s.get('classification_weight',1.)*masked_mean(ce,weights)+s.get('cdf_weight',.2)*masked_mean(emd,weights)
    if 'raw_y' in output:
        loss=loss+.2*masked_mean(F.relu(output['raw_y'][:,0]-output['raw_y'][:,1]),both)
    return loss


def distillation(student,teacher,y,valid):
    gate=torch.exp(-(teacher['y']-y).abs().detach()/8)*valid
    coord=masked_mean(robust(student['y'],teacher['y'].detach()),gate)
    sp=student['regions'].float().clamp(1e-5,1-1e-5)
    tp=F.interpolate(teacher['regions'].detach().float(),size=sp.shape[-2:],mode='bilinear',align_corners=False).clamp(1e-5,1-1e-5)
    region_gate=F.interpolate(gate,size=sp.shape[-1],mode='linear',align_corners=False)[:,:,None,:].expand_as(sp)
    kl=tp*(tp.log()-sp.log())+(1-tp)*((1-tp).log()-(1-sp).log())
    return coord+.5*masked_mean(kl,region_gate)
