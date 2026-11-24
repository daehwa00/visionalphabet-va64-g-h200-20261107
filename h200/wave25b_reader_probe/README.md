# Wave 25B reader verification

기준은 Wave 21C 커밋 `039440f2c0a9a9d006948fcbbf45dd5c96e36efb`이다. 기본값과 `LNET_READER_W25B=off`는 기존 디스패치를 실행한다. 명시적으로 `auto` 또는 `on`을 지정하면 SM 8.6과 SM 9.0에서 후보를 선택한다. `LNET_READER_VEC` 강제 설정은 기존 경로를 유지한다.

## 오프라인 리소스 확인

격리 클론 루트에서 실행한다.

```bash
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe
mkdir -p /tmp/t90-refs /tmp/t90-plans
git show 039440f2:src/lnet/factorized_reader_kernels.py > /tmp/t90-refs/wave21.py
git show origin/wave20b-reader-p1nr4:src/lnet/factorized_reader_kernels.py > /tmp/t90-refs/wave20.py
python h200/wave25b_reader_probe/replay.py --refs /tmp/t90-refs --out /tmp/t90-plans
python h200/wave25b_reader_probe/replay.py --matrix --out /tmp/t90-matrix
python h200/wave25b_reader_probe/replay.py --matrix --refs /tmp/t90-refs --arms wave21 --out /tmp/t90-matrix-wave21
python h200/wave25b_reader_probe/weight_resources.py --out /tmp/t90-weight-resources.jsonl
```

`replay.py`는 GPU 속성과 드라이버 호출을 대체하고 실제 Python 런처에서 선택한 모듈을 NVRTC와 `ptxas`로 빌드한다. 트래픽 명령 수는 SASS에 정적으로 나타나는 instruction 수다. `weight_resources.py`는 개별 실험 구성을 따로 어셈블한다.

## RTX 3080 Ti

격리 클론을 `gunsoo:/tmp/va-t90-gunsoo`에 전송한 뒤 해당 호스트에서 다음을 실행한다. 스크립트가 각 GPU 명령에 공용 잠금을 적용한다.

```bash
cd /tmp/va-t90-gunsoo
bash h200/wave25b_reader_probe/gunsoo.sh ablate
bash h200/wave25b_reader_probe/gunsoo.sh bench
probe=h200/wave25b_reader_probe/gunsoo.sh
bash "$probe" train-pair controlled-1 off auto --no-cudnn-benchmark
bash "$probe" train-pair controlled-2 auto off --no-cudnn-benchmark
bash "$probe" train-pair controlled-3 off auto --no-cudnn-benchmark
bash "$probe" train-pair controlled-4 auto off --no-cudnn-benchmark
bash "$probe" train-pair benchmark-1 off auto
bash "$probe" train-pair benchmark-2 auto off
bash h200/wave25b_reader_probe/gunsoo.sh verify
```

CUDA Graph는 순환 순서에서 CUDA 이벤트 쌍당 8번 같은 커널을 실행한다. 논리적 GB/s는 해당 런치의 텐서 바이트를 실행시간으로 나눈 값이다. `train-pair`는 공용 GPU 잠금을 한 쌍 동안 유지한다. 각 arm 전에 25초를 쉬고, 같은 시드의 독립 프로세스를 실행하며 온도와 클록을 1초마다 기록한다.
