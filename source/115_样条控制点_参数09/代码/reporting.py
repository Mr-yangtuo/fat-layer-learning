"""Refresh project/scheme README status without creating root management files."""
import fcntl
import json
import os
import re
from datetime import datetime
from pathlib import Path

SCHEME=Path(__file__).resolve().parents[1]
ROOT=SCHEME.parent
START,END='<!-- STATUS:BEGIN -->','<!-- STATUS:END -->'
ROLES=('teacher','mobilenet_v3_large','mobilenet_v3_small')
ZH={'queued':'未开始','running':'训练中','completed':'完成','failed':'失败','paused':'已暂停','waiting':'等待资源'}


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def checkpoint_name(sid,split,role):
    tail='teacher_resnet18' if role=='teacher' else role+'_distilled'
    return f'{sid}_{split}_{tail}_best.pth'


def replace_section(path,section):
    text=path.read_text()
    if text.count(START)!=1 or text.count(END)!=1:raise ValueError(f'Invalid README markers: {path}')
    before,rest=text.split(START);_,after=rest.split(END)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(before+START+'\n'+section+'\n'+END+after);temporary.replace(path)


def refresh():
    # Lock the directory inode; no .lock or management folder is left in the root.
    directory_fd=os.open(ROOT,os.O_RDONLY)
    try:
        fcntl.flock(directory_fd,fcntl.LOCK_EX)
        summary=['更新时间：'+now(),'','| 方案 | 完成任务 | 当前状态 | train90 真皮下界 MAE | train90 脂肪下界 MAE |',
                 '| --- | --- | --- | --- | --- |']
        for folder in sorted(ROOT.iterdir()):
            if not folder.is_dir() or not re.match(r'^\d{3,}_',folder.name) or not (folder/'README.md').exists():continue
            rows=['| 划分 | 模型 | 状态 | 已完成轮次 | 真皮下界 MAE | 脂肪下界 MAE | 权重 | 大小 MB |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
            count=0;active=[];teacher_metrics={}
            for split in ('train90','train100'):
                for role in ROLES:
                    state_file=folder/'训练记录'/split/role/'status.json'
                    state=json.loads(state_file.read_text()) if state_file.exists() else {'state':'queued'}
                    status=state.get('state','queued');count+=status=='completed'
                    if status in ('running','paused','failed','waiting'):active.append(f'{split}/{role} {ZH[status]}')
                    metric_file=folder/'训练记录/指标'/f'{split}_{role}_best.json'
                    metrics=json.loads(metric_file.read_text()) if metric_file.exists() else {}
                    if split=='train90' and role=='teacher':teacher_metrics=metrics
                    scores=[f'{metrics[key]:.4f}' if key in metrics else '—' for key in ('dermis_mae','fat_mae')]
                    name=checkpoint_name(folder.name,split,role)
                    path=folder/'模型'/split/name
                    link=f'[{name}](模型/{split}/{name})' if path.is_file() else '暂无'
                    rows.append(f'| {split} | {role} | {ZH.get(status,status)} | {state.get("epoch",0)} | '+
                                ' | '.join(scores)+f' | {link} | '+(f'{path.stat().st_size/1e6:.2f}' if path.is_file() else '—')+' |')
            rows+=['','MAE 为同一个已保存权重的逐线误差，单位是 256×256 输入像素；越小越好。',
                   'train90 为分组验证，train100 为训练集复评，二者分开解释。']
            replace_section(folder/'README.md','\n'.join(rows))
            vals=[f'{teacher_metrics[key]:.4f}' if key in teacher_metrics else '—' for key in ('dermis_mae','fat_mae')]
            summary.append(f'| [{folder.name}]({folder.name}/README.md) | {count}/6 | '+
                           ('；'.join(active) or ('完成' if count==6 else '尚有未训练任务'))+' | '+' | '.join(vals)+' |')
        summary+=['','上表 MAE 对应各方案已保存的 train90 教师权重；未完成训练的数值仅为阶段性结果。']
        replace_section(ROOT/'README.md','\n'.join(summary))
    finally:
        os.close(directory_fd)


if __name__=='__main__':refresh()
