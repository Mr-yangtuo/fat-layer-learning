"""Offline teacher/student training with explicit split isolation and PTH-only weights."""
import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import build_model
from data import BoundaryDataset
from losses import supervised, distillation

SCHEME = Path(__file__).resolve().parents[1]
from reporting import atomic_json, checkpoint_name, now, refresh


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def save_checkpoint(path, payload):
    temporary = path.with_name(path.stem + '.tmp.pth')
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total = np.zeros(2); count = np.zeros(2); original_total = np.zeros(2)
    accuracy = np.zeros(3); errors = []; failures = 0; images = 0
    thickness_total = 0.; thickness_count = 0
    for batch in loader:
        image = batch['image'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(image)
        error = (output['y'].float()-y).abs()
        if not torch.isfinite(error).all():
            raise RuntimeError('Nonfinite prediction in evaluation')
        scale = batch['original_scale'].to(device)[:,None,None]
        total += (error*valid).sum((0,2)).cpu().numpy()
        original_total += (error*valid*scale).sum((0,2)).cpu().numpy()
        count += valid.sum((0,2)).cpu().numpy()
        selected = error[valid]
        accuracy += np.array([(selected <= threshold).sum().item() for threshold in (2,3,5)])
        errors.append(selected.cpu())
        sample_mae = (error*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        failures += (sample_mae.max(1).values > 10).sum().item()
        images += len(image)
        both = valid.all(1)
        thickness = ((output['y'][:,1]-output['y'][:,0])-(y[:,1]-y[:,0])).abs()
        thickness_total += (thickness*both).sum().item(); thickness_count += both.sum().item()
    per_line = total / np.maximum(count, 1)
    return dict(mae=float(per_line.mean()), dermis_mae=float(per_line[0]), fat_mae=float(per_line[1]),
                dermis_mae_original_px=float(original_total[0]/max(count[0],1)),
                fat_mae_original_px=float(original_total[1]/max(count[1],1)),
                accuracy_2=float(accuracy[0]/max(count.sum(),1)), accuracy_3=float(accuracy[1]/max(count.sum(),1)),
                accuracy_5=float(accuracy[2]/max(count.sum(),1)),
                p95_vertical_error=float(torch.cat(errors).quantile(.95)),
                failure_rate_mae_gt10=failures/max(images,1),
                thickness_mae=thickness_total/max(thickness_count,1), images=images,
                valid_columns_per_line=count.tolist())


def train(args):
    config = json.loads((SCHEME/'代码/config.json').read_text())
    split, role = args.split, args.role
    role_backbone = 'resnet18' if role == 'teacher' else role
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('This project is authorized for physical GPU0 only; require CUDA_VISIBLE_DEVICES=0')
    torch.cuda.set_per_process_memory_fraction(0.25, 0)
    device = torch.device('cuda:0')
    epochs = config['teacher_epochs'] if role == 'teacher' else config['student_epochs']
    logs = SCHEME/'训练记录'/split/role; logs.mkdir(parents=True,exist_ok=True)
    models = SCHEME/'模型'/split;models.mkdir(parents=True,exist_ok=True)
    best_file = models/checkpoint_name(SCHEME.name,split,role)
    last_file = models/best_file.name.replace('_best.pth','_last.pth')
    metric_file = SCHEME/'训练记录/指标'/f'{split}_{role}.jsonl'
    status = dict(state='running',split=split,role=role,backbone=role_backbone,epoch=0,epochs=epochs,
                  started_at=now(),pid=os.getpid(),gpu=0, best_mae=None)
    def publish(**updates):
        status.update(updates);status['updated_at']=now()
        atomic_json(logs/'status.json',status);refresh()
    publish()
    try:
        audit=json.loads((SCHEME/'代码/data_index.json').read_text())['audit']
        train_data=BoundaryDataset(SCHEME/'代码/data_index.json',split,True,settings=config.get('augmentation'))
        eval_data=BoundaryDataset(SCHEME/'代码/data_index.json',split,False)
        if split=='train90':
            tr_sources={train_data.items[i]['image'] for i in train_data.indices}
            ev_sources={eval_data.items[i]['image'] for i in eval_data.indices}
            assert not tr_sources & ev_sources
        generator=torch.Generator().manual_seed(seed)
        loader=DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,
                          num_workers=config['workers'],pin_memory=True,persistent_workers=True,
                          worker_init_fn=seed_worker,generator=generator,prefetch_factor=2)
        val_loader=DataLoader(eval_data,batch_size=config['batch_size'],shuffle=False,
                              num_workers=2,pin_memory=True,persistent_workers=True,worker_init_fn=seed_worker)
        model=build_model(config['method'],role_backbone,SCHEME/'模型/初始化权重',config.get('architecture')).to(device)
        size_buffer=io.BytesIO()
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()}},size_buffer)
        estimated_bytes=size_buffer.tell();del size_buffer
        if role!='teacher' and estimated_bytes>config.get('deployment_limit_bytes',20_000_000)-100_000:
            raise RuntimeError(f'Student model exceeds deployment budget: {estimated_bytes} bytes')
        publish(estimated_checkpoint_bytes=estimated_bytes,deployment_target=role!='teacher')
        teacher=None
        teacher_file=None
        teacher_hash=None
        if role!='teacher':
            teacher_file=models/checkpoint_name(SCHEME.name,split,'teacher')
            teacher_status=json.loads((SCHEME/'训练记录'/split/'teacher/status.json').read_text())
            if teacher_status['state']!='completed':
                raise RuntimeError('Teacher training not completed')
            saved=torch.load(teacher_file,map_location='cpu',weights_only=True)
            assert saved['split']==split and saved['scheme']==SCHEME.name and saved['role']=='teacher'
            assert saved['data_fingerprint']==audit['annotation_manifest_sha256']
            teacher=build_model(config['method'],'resnet18',settings=config.get('architecture')).to(device)
            teacher.load_state_dict(saved['model'],strict=True)
            teacher.eval().requires_grad_(False)
            teacher_hash=hashlib.sha256(teacher_file.read_bytes()).hexdigest()
            del saved
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-4)
        def schedule(epoch):
            if epoch<5:return (epoch+1)/5
            return .05+.95*.5*(1+math.cos(math.pi*(epoch-5)/max(epochs-5,1)))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
        code_digest=hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob('*.py')):
            code_digest.update(path.name.encode());code_digest.update(path.read_bytes())
        code_hash=code_digest.hexdigest()
        start,best,best_epoch=1,float('inf'),0
        best_score=float('inf')
        if last_file.exists():
            saved=torch.load(last_file,map_location=device,weights_only=False)
            assert saved['config']==config and saved['source_hash']==code_hash, 'Changed configuration/source; use a new scheme'
            assert saved['teacher_sha256']==teacher_hash
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            start=saved['epoch']+1;best=saved['best_mae'];best_epoch=saved['best_epoch']
            best_score=saved.get('best_score',best)
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy'])
            torch.set_rng_state(saved['rng_torch'].cpu());torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
            generator.set_state(saved['rng_loader'].cpu())
            del saved
        print(json.dumps(dict(event='started',scheme=SCHEME.name,split=split,role=role,train_views=len(train_data),
                              eval_originals=len(eval_data),epochs=epochs,start_epoch=start,gpu=torch.cuda.get_device_name(),
                              train100_evaluation_overlaps=split=='train100',parameters=sum(p.numel() for p in model.parameters())),ensure_ascii=False),flush=True)
        metadata=dict(format_version=1,scheme=SCHEME.name,split=split,role=role,backbone=role_backbone,
                      method=config['method'],config=config,labels=audit['labels'],input_size=[256,256],
                       architecture=config.get('architecture',{}),deployment_target=role!='teacher',
                       method_family=config['method_family'],trial=config['trial'],
                      input_preprocessing='grayscale [0,1], repeat 3 channels, ImageNet mean/std inside model',
                      output='B x 2 x W ordered boundary y coordinates in input pixels',
                      evaluation='independent grouped validation' if split=='train90' else 'training-set reevaluation',
                      data_fingerprint=audit['annotation_manifest_sha256'],source_hash=code_hash,
                      teacher_path=str(teacher_file) if teacher_file else None,teacher_sha256=teacher_hash)
        for epoch in range(start,epochs+1):
            epoch_start=time.time();model.train();running=0.;batches=0
            optimizer.zero_grad(set_to_none=True)
            for step,batch in enumerate(loader):
                image=batch['image'].to(device,non_blocking=True)
                y=batch['y'].to(device,non_blocking=True)
                valid=batch['valid'].to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(image)
                    loss=supervised(output,y,valid,config.get('loss'))
                    if teacher is not None:
                        with torch.no_grad():teacher_output=teacher(image)
                        loss=loss+config['distill_weight']*min(epoch/10,1)*distillation(output,teacher_output,y,valid)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss epoch={epoch} step={step}')
                accumulation=config['accumulation']
                group_start=(step//accumulation)*accumulation
                group_size=min(accumulation,len(loader)-group_start)
                (loss/group_size).backward()
                if (step+1)%accumulation==0 or step+1==len(loader):
                    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    if not torch.isfinite(grad_norm):raise RuntimeError('Nonfinite gradient')
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                running+=loss.item();batches+=1
                if step==0 or (step+1)%100==0:
                    print(f'epoch={epoch}/{epochs} batch={step+1}/{len(loader)} loss={running/batches:.5f} gpu_allocated_mb={torch.cuda.memory_allocated()/2**20:.0f}',flush=True)
            metrics=evaluate(model,val_loader,device)
            scheduler.step()
            reference=config.get('selection_reference',[1.,1.])
            ratios=[metrics['dermis_mae']/reference[0],metrics['fat_mae']/reference[1]]
            score=max(ratios)+.05*sum(ratios)/2 if config.get('selection')=='balanced_lines' else metrics['mae']
            record=dict(epoch=epoch,train_loss=running/batches,seconds=time.time()-epoch_start,lr=optimizer.param_groups[0]['lr'],selection_score=score,**metrics)
            with metric_file.open('a') as file:file.write(json.dumps(record)+'\n')
            if score<best_score:
                best=metrics['mae'];best_score=score;best_epoch=epoch
                payload=dict(metadata,epoch=epoch,metrics=metrics,selection_score=score,model={key:value.detach().cpu() for key,value in model.state_dict().items()})
                save_checkpoint(best_file,payload)
                atomic_json(SCHEME/'训练记录/指标'/f'{split}_{role}_best.json',record)
            last=dict(metadata,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                      best_mae=best,best_score=best_score,best_epoch=best_epoch,rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                      rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),rng_loader=generator.get_state())
            save_checkpoint(last_file,last)
            publish(epoch=epoch,best_mae=best,best_score=best_score,best_epoch=best_epoch,last_metrics=record)
            print(json.dumps(dict(event='epoch_completed',best_mae=best,best_epoch=best_epoch,**record)),flush=True)
            if epoch>=config['min_epochs'] and epoch-best_epoch>=config['patience']:
                print('Early stop: validation/reevaluation metric did not improve.',flush=True);break
        saved=torch.load(best_file,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True)
        final_metrics=evaluate(model,val_loader,device)
        final_metrics['checkpoint_bytes']=best_file.stat().st_size
        if role!='teacher' and best_file.stat().st_size>config.get('deployment_limit_bytes',20_000_000):
            raise RuntimeError('Saved student PTH exceeds 20 MB deployment budget')
        atomic_json(SCHEME/'训练记录/指标'/f'{split}_{role}_final.json',dict(checkpoint=str(best_file),**final_metrics))
        # Deterministic sample overlay, saved only for final model.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sample=next(iter(val_loader));model.eval()
        with torch.inference_mode():pred=model(sample['image'][:6].to(device))['y'].cpu()
        fig,axes=plt.subplots(2,3,figsize=(10,7))
        for i,ax in enumerate(axes.flat):
            if i>=len(pred):ax.axis('off');continue
            ax.imshow(sample['image'][i,0],cmap='gray')
            for line,color in enumerate(['lime','orange']):
                truth=sample['y'][i,line].numpy().copy();truth[~sample['valid'][i,line].numpy()]=np.nan
                ax.plot(truth,color=color,linewidth=1);ax.plot(pred[i,line],color=color,linestyle='--',linewidth=1)
            ax.axis('off')
        fig.suptitle('Solid: GT; dashed: prediction');fig.tight_layout()
        fig.savefig(SCHEME/'训练记录/预测'/f'{split}_{role}.png',dpi=120);plt.close(fig)
        # Final best must be reloadable before deleting this job's resumable checkpoint.
        last_file.unlink(missing_ok=True)
        publish(state='completed',finished_at=now(),checkpoint=str(best_file),checkpoint_sha256=hashlib.sha256(best_file.read_bytes()).hexdigest(),final_metrics=final_metrics)
        print('COMPLETED',best_file,flush=True)
    except BaseException as error:
        publish(state='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train90','train100'],required=True)
    parser.add_argument('--role',choices=['teacher','mobilenet_v3_large','mobilenet_v3_small'],required=True)
    train(parser.parse_args())
