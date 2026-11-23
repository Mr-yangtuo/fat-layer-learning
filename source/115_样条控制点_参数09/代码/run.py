"""Explicit per-scheme training entry; no background queue or automatic search."""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEME=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['train90','train100'],required=True)
    parser.add_argument('--role',choices=['teacher','mobilenet_v3_large','mobilenet_v3_small'],required=True)
    args=parser.parse_args()
    logs=SCHEME/'训练记录'/args.split/args.role
    status=logs/'status.json'
    if status.exists() and json.loads(status.read_text()).get('state')=='completed':
        print('该任务已完成；保留已有权重。若需重新实验，请创建新方案。');return
    logs.mkdir(parents=True,exist_ok=True)
    child=None
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    # The runtime creates temporary caches only for this process and removes them on exit.
    with tempfile.TemporaryDirectory(prefix='.运行临时_',dir=SCHEME/'训练记录') as temp:
        # Linux AF_UNIX sockets have a short byte limit. The proc alias keeps
        # multiprocessing paths short while all files stay inside this scheme.
        # The trainer and its data workers share cwd=SCHEME throughout the run.
        socket_temp=str(Path('/proc/self/cwd')/'训练记录'/Path(temp).name)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TMPDIR=socket_temp,TORCH_HOME=temp,
                   HF_HOME=temp,HF_HUB_OFFLINE='1',MPLCONFIGDIR=temp,XDG_CACHE_HOME=temp,
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        with (logs/'training.log').open('a',buffering=1) as log:
            try:
                child=subprocess.Popen([sys.executable,'-B','-u',str(SCHEME/'代码/train.py'),
                                        '--split',args.split,'--role',args.role],cwd=SCHEME,env=env,
                                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                print(f'GPU0 训练进程 {child.pid}；日志：{logs / "training.log"}',flush=True)
                code=child.wait()
            except KeyboardInterrupt:
                if child is not None and child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
                from reporting import atomic_json,now,refresh
                value=json.loads(status.read_text()) if status.exists() else {}
                value.update(state='paused',paused_at=now(),pause_reason='用户停止')
                atomic_json(status,value);refresh();return
    raise SystemExit(code)


if __name__=='__main__':main()
