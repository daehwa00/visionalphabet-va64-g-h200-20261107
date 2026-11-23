# Wave 24B reader verification

기준은 Wave 21C 커밋 `039440f2c0a9a9d006948fcbbf45dd5c96e36efb`이다. `LNET_READER_W24B=off`는 기존 디스패치를 실행한다. `on`은 실험 경로를 선택한다. `LNET_READER_VEC` 강제 설정은 기존 경로를 유지한다.

## 오프라인 리소스 확인

격리 클론 루트에서 실행한다.

```bash
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe
mkdir -p /tmp/t87-refs /tmp/t87-plans
git show 039440f2:src/lnet/factorized_reader_kernels.py > /tmp/t87-refs/wave21.py
git show origin/wave20b-reader-p1nr4:src/lnet/factorized_reader_kernels.py > /tmp/t87-refs/wave20.py
python h200/wave24b_reader_probe/replay.py --refs /tmp/t87-refs --out /tmp/t87-plans
python h200/wave24b_reader_probe/replay.py --matrix --out /tmp/t87-matrix
python h200/wave24b_reader_probe/weight_resources.py --out /tmp/t87-weight-resources.jsonl
```

`replay.py`는 GPU 속성과 드라이버 호출을 대체하고 실제 Python 런처에서 선택한 모듈을 NVRTC와 `ptxas`로 빌드한다. 트래픽 명령 수는 SASS에 정적으로 나타나는 instruction 수다. `weight_resources.py`는 개별 실험 구성을 따로 어셈블한다.

## RTX 3080 Ti

격리 클론을 `gunsoo:/tmp/va-t87-gunsoo`에 전송한 뒤 해당 호스트에서 다음을 실행한다. 스크립트가 각 GPU 명령에 공용 잠금을 적용한다.

```bash
cd /tmp/va-t87-gunsoo
bash h200/wave24b_reader_probe/gunsoo.sh ablate
bash h200/wave24b_reader_probe/gunsoo.sh bench
bash h200/wave24b_reader_probe/gunsoo.sh train
bash h200/wave24b_reader_probe/gunsoo.sh verify
```

CUDA Graph는 순환 순서에서 CUDA 이벤트 쌍당 8번 같은 커널을 실행한다. 논리적 GB/s는 해당 런치의 텐서 바이트를 실행시간으로 나눈 값이다. `train`은 같은 시드의 독립 프로세스에서 두 실행 순서와 cuDNN 벤치마크를 끈 메모리 비교를 실행한다.
