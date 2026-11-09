"""Run the exact shell queue branch with a disposable fake trainer."""
from pathlib import Path
import os
import subprocess
import pytest


@pytest.mark.parametrize('failure,stop,expected',[(False,False,[1,2]),(True,False,[1]),(False,True,[1])])
def test_seed_queue_order_and_stop_on_first_failure(tmp_path,failure,stop,expected):
 source=(Path(__file__).resolve().parents[1]/'h200/va64/run.sh').read_text()
 marker='if [[ "$MODE" == seeds12 ]]; then\n  unset'
 branch='  unset'+source.split(marker,1)[1].split('\nfi',1)[0]
 py=tmp_path/'trainer';py.write_text('''#!/usr/bin/env python3
import sys,os,pathlib,json
seed=int(sys.argv[sys.argv.index('--seed')+1]);root=pathlib.Path(os.environ['TEST_ROOT'])
with (root/'calls').open('a') as f:f.write(str(seed)+'\\n')
assert os.environ['VA64_G_JOB']==f'va64-g-seed{seed}'
assert sys.argv[sys.argv.index('--batch-size')+1]=='128'
assert '--gpu-mixup' in sys.argv
if os.environ['TEST_STOP']=='1':(root/'STOP').touch()
if os.environ['TEST_FAIL']=='1':sys.exit(1)
''');py.chmod(0o700)
 script='set -euo pipefail\ncheck_stop() { [[ ! -e "$RUN_ROOT/STOP" ]] || exit 130; }\n'+branch
 env={**os.environ,'TEST_ROOT':str(tmp_path),'TEST_FAIL':str(int(failure)),'TEST_STOP':str(int(stop)),'RUN_ROOT':str(tmp_path),'DATA_ROOT':'/unused','PY':str(py)}
 result=subprocess.run(['bash','-c',script],env=env,capture_output=True,text=True)
 assert list(map(int,(tmp_path/'calls').read_text().splitlines()))==expected
 assert result.returncode==(1 if failure else 130 if stop else 0)
