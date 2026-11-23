"""GPU0 sequential teacher/distillation jobs and ongoing search in the documented parameter space."""
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from reporting import atomic_json,checkpoint_name,now,refresh

ROOT=Path(__file__).resolve().parents[2]
ROLES=('teacher','mobilenet_v3_large','mobilenet_v3_small')


def schemes():
    return sorted([p for p in ROOT.iterdir() if p.is_dir() and re.match(r'^\d{3,}_',p.name)
                   and (p/'代码/config.json').exists()],key=lambda p:int(p.name.split('_')[0]))


def state(folder,split,role):
    path=folder/'训练记录'/split/role/'status.json'
    return json.loads(path.read_text()) if path.exists() else {'state':'queued','attempts':0}


def jobs():
    folders=schemes()
    return ([(p,s,'teacher') for s in ('train90','train100') for p in folders]+
            [(p,s,r) for s in ('train90','train100') for p in folders for r in ROLES[1:]])


def completed_teachers():
    records=[]
    for p in schemes():
        file=p/'训练记录/指标/train90_teacher_best.json'
        if state(p,'train90','teacher')['state']=='completed' and file.exists():
            records.append({'folder':p,'metrics':json.loads(file.read_text()),
                            'config':json.loads((p/'代码/config.json').read_text())})
    return records


def frontier(records):
    return [r for r in records if not any(
        all(s['metrics'][key]<=r['metrics'][key] for key in ('dermis_mae','fat_mae')) and
        any(s['metrics'][key]<r['metrics'][key] for key in ('dermis_mae','fat_mae')) for s in records)]


def experiment_key(config):
    keys=['method','lr','distill_weight','loss','augmentation','selection','selection_reference']
    return hashlib.sha256(json.dumps({k:config.get(k) for k in keys},sort_keys=True).encode()).hexdigest()


def describe(folder,config,parent):
    method='有序双曲线回归：多尺度特征、横向卷积及两层 Transformer，预测三个非负分层比例并累加为两条线。' if config['method']=='ordered_curves' else '嵌套区域边界：多尺度特征预测两条边界以下的嵌套组织区域概率，再沿高度积分得到两条线。'
    return f'''# {folder.name}

本方案是既有结构的参数/增强变体，参数父方案为 `{parent['folder'].name}`。
{method}
教师采用 ResNet18；学生为 MobileNetV3-Large 和 Small，均配合本方案解码结构。
各模型从 ImageNet 初始化权重开始，不继承父方案训练权重。新方案是否更好必须看训练后的逐线结果。

<!-- STATUS:BEGIN -->
尚未训练。
<!-- STATUS:END -->

## 本次变化及配置

父方案 train90 教师的真皮下界 MAE 为 {parent['metrics']['dermis_mae']:.4f}，脂肪下界 MAE 为 {parent['metrics']['fat_mae']:.4f}。
父方案从同结构的逐线 Pareto 候选中选择；本次调整学习率、损失权重、蒸馏权重和温和噪声概率。
具体配置完整写在 `代码/config.json`：

```json
{json.dumps(config,ensure_ascii=False,indent=2)}
```

`line_weights` 分别对应真皮层下界和脂肪层下界的坐标误差；`coord_beta` 为坐标 SmoothL1 转折参数。
`slope_weight`、`thickness_weight`、`region_weight` 分别是斜率、厚度和区域 BCE 权重；纵向单调惩罚固定为 0.5。
`noise_probability` 为散斑概率，`blur_probability` 为模糊概率，`ridge_probability` 为避开真值边界的伪亮条纹概率。
配置差异是待验证假设，不能把新建方案视为已经改进。

## 数据、裁剪与验证

固定 `/home/zlg/0.减脂数据集/无垫/腹部/`，共 8,300 张有效原图。
train90 使用 7,470 张原图（11,617 个含裁剪视图的样本）训练，830 张按目录组独立验证；
train100 使用全部 13,277 个视图，830 张仅作训练集复评。
`代码/data_index.json` 保持相同划分和样本顺序。直接读原图，不保存大型磁盘图像缓存。

强制保留 516:594 比例的原图+安全裁剪副本，以及原有三个在线裁剪/压缩函数。
在线裁剪三选一概率固定 0.3，图像、坐标与有效掩码同步；裁掉整条目标线的副本不加入训练。
水平翻转、温和亮度/对比度/gamma、低分辨率和纵向增益保持原实现，其余概率见上表。

## 训练与蒸馏

GPU0 顺序执行，每种划分一个教师和两种学生，共 6 项任务。教师最多 {config['teacher_epochs']} 轮，学生最多 {config['student_epochs']} 轮，
至少 {config['min_epochs']} 轮后连续 {config['patience']} 轮未改善可早停。微批量 {config['batch_size']}、累积 {config['accumulation']} 次，BF16。
教师冻结并用 eval 模式与学生读取同一增强图像，蒸馏坐标和区域分布；教师偏差大的训练点降低蒸馏权重。
train90 学生只使用同方案 train90 教师。所有模型只保存 `.pth`，训练中 last 用于恢复，最终重载验证成功后只保留 best。

保存权重时兼顾两条线：分别除以固定参考 MAE 后，最小化“较差的相对误差 + 0.05×两线相对误差均值”。
原始逐线 MAE 始终同时报告；这一选择规则不保证每个新方案都能同时改善两条线。
train90 是持续调参的开发验证集，不冒充从未参与选择的最终测试集。

## 文件和运行

本目录仅有 README、代码、模型和训练记录。代码、配置和固定划分都在本方案中；模型分 train90、train100、初始化权重三处。
单独运行示例：

```bash
/home/zlg/anaconda3/envs/torch/bin/python -B 代码/run.py --split train90 --role teacher
```

角色还可设为 `mobilenet_v3_large` 或 `mobilenet_v3_small`，划分可换为 train100。
日志、逐轮指标和最终预测图在 `训练记录`。临时缓存仅在进程运行期间创建，正常退出/中断时清理。
`代码/reporting.py` 自动更新两级 README。结构参考思想见最初两个方案的详细说明；本方案没有引入新的外部仓库实现。

## 更新记录

- {now()}：持续搜索根据已有逐线验证结果创建本参数变体；尚无本方案训练结论。
'''


def propose_next_pair():
    records=completed_teachers()
    if not records:raise RuntimeError('尚无完成的 train90 教师，不能根据结果生成下一轮；保留记录等待处理。')
    existing={experiment_key(json.loads((p/'代码/config.json').read_text())) for p in schemes()}
    created=[]
    for method,label in [('ordered_curves','曲线调优'),('ordinal_regions','区域调优')]:
        candidates=[r for r in records if r['config']['method']==method]
        if not candidates:continue
        index=max(int(p.name.split('_')[0]) for p in schemes())+1
        rng=random.Random(20260925+index*7919)
        parent=rng.choice(frontier(candidates))
        for attempt in range(100):
            config=json.loads(json.dumps(parent['config']))
            config.update(scheme=f'{index:03d}_{label}',parent_scheme=parent['folder'].name,
                          lr=round(min(.0007,max(.00005,parent['config']['lr']*math.exp(rng.uniform(-.5,.5)))),9),
                          distill_weight=round(rng.uniform(.25,.8),3))
            config['loss']={'line_weights':[round(rng.uniform(.75,1.5),3),round(rng.uniform(.75,1.5),3)],
                            'coord_beta':rng.choice([1.,2.,3.]),'slope_weight':rng.choice([.1,.25,.4]),
                            'thickness_weight':rng.choice([.05,.15,.25]),'region_weight':rng.choice([1.,2.,3.])}
            config['augmentation']={'noise_probability':rng.choice([.05,.1,.15]),
                                    'blur_probability':rng.choice([0.,.05,.1]),'ridge_probability':rng.choice([0.,.1,.15])}
            if experiment_key(config) not in existing:break
        else:raise RuntimeError('Unable to generate a distinct candidate')
        folder=ROOT/config['scheme'];folder.mkdir()
        try:
            code=folder/'代码';code.mkdir()
            for source in (parent['folder']/'代码').glob('*.py'):shutil.copy2(source,code/source.name)
            os.link(parent['folder']/'代码/data_index.json',code/'data_index.json')
            atomic_json(code/'config.json',config)
            for split in ('train90','train100'):
                (folder/'模型'/split).mkdir(parents=True)
                for role in ROLES:(folder/'训练记录'/split/role).mkdir(parents=True)
            (folder/'模型/初始化权重').mkdir()
            for source in (parent['folder']/'模型/初始化权重').glob('*.pth'):
                os.link(source,folder/'模型/初始化权重'/source.name)
            for name in ('指标','预测'):(folder/'训练记录'/name).mkdir()
            atomic_json(folder/'训练记录/方案来源.json',{'created_at':now(),'parent':parent['folder'].name,
                         'parent_metrics':parent['metrics'],'experiment_key':experiment_key(config),
                         'search_scope':'two implemented structures; learning rate, loss weights, non-crop augmentation, distillation strength',
                         'from_scratch':True,'copied_trained_weights':False})
            (folder/'README.md').write_text(describe(folder,config,parent))
        except BaseException:
            # Only remove an incomplete, newly created candidate with no trained model.
            shutil.rmtree(folder);raise
        existing.add(experiment_key(config));created.append(folder.name)
    refresh();return created


def main():
    owner=schemes()[0]
    # Reuse a directory inode for process locking, so no root lock file/folder is needed.
    lock_fd=os.open(owner/'模型',os.O_RDONLY)
    try:fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('已有本项目批量训练进程，拒绝重复启动。')
    record_file=owner/'训练记录/批量训练状态.json'
    child=None
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        while True:
            progressed=False
            for folder,split,role in jobs():
                current=state(folder,split,role)
                if current['state']=='completed' or (current['state']=='failed' and current.get('attempts',0)>=2):continue
                status_file=folder/'训练记录'/split/role/'status.json'
                if role!='teacher' and state(folder,split,'teacher')['state']!='completed':
                    teacher=state(folder,split,'teacher')
                    permanent=teacher.get('state')=='failed' and teacher.get('attempts',0)>=2
                    atomic_json(status_file,{'state':'failed' if permanent else 'queued','attempts':2 if permanent else 0,
                                             'error':'对应教师未完成，不执行蒸馏。','updated_at':now()})
                    refresh();continue
                while True:
                    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free>=6144 and shutil.disk_usage(ROOT).free>=10*1024**3:break
                    atomic_json(record_file,{'state':'waiting','pid':os.getpid(),'gpu':0,'gpu_free_mib':free,'updated_at':now()})
                    print(now(),'等待 GPU0 显存/磁盘空间',flush=True);time.sleep(30)
                attempt=current.get('attempts',0)+1
                atomic_json(record_file,{'state':'running','pid':os.getpid(),'scheme':folder.name,'split':split,'role':role,'attempt':attempt,'updated_at':now()})
                print(now(),'START',folder.name,split,role,'attempt',attempt,flush=True)
                child=subprocess.Popen([sys.executable,'-B','-u',str(folder/'代码/run.py'),'--split',split,'--role',role],
                                       cwd=folder,start_new_session=True)
                code=child.wait();child=None;progressed=True
                result=state(folder,split,role);result['attempts']=attempt
                if result.get('state')=='paused':raise KeyboardInterrupt
                if code or result.get('state')!='completed':
                    result.update(state='failed',error=f'退出码 {code}，详见本任务 training.log')
                atomic_json(status_file,result);refresh()
                print(now(),'END',folder.name,split,role,code,flush=True)
            pending=[j for j in jobs() if state(*j)['state']!='completed' and state(*j).get('attempts',0)<2]
            if pending:
                if not progressed:time.sleep(30)
                continue
            try:
                created=propose_next_pair()
                print(now(),'NEW SCHEMES',created,flush=True)
                atomic_json(record_file,{'state':'searching','pid':os.getpid(),'created':created,'updated_at':now()})
            except Exception as error:
                atomic_json(record_file,{'state':'waiting','pid':os.getpid(),'error':str(error),'updated_at':now()})
                print(now(),'SEARCH WAIT',repr(error),flush=True);time.sleep(60)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
        atomic_json(record_file,{'state':'paused','pid':os.getpid(),'reason':'用户停止','updated_at':now()})
        print(now(),'用户停止，未启动后续任务。',flush=True)
    finally:os.close(lock_fd)


if __name__=='__main__':main()
