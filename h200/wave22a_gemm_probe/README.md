# Wave 22A 진단 재현

Wave 17B의 P1/NR4 학습 스텝에서 CUDA 커널과 ATen 연산을 연결하고, 분리된 프로세스에서 학습 스텝 후보를 비교한다. 모델과 대표 배치 생성은 `h200/liverms_v2_opt/bench_h200.py`를 사용한다. 기본 배치는 128, 시드는 501이며 합성 224×224 channels-last 입력에서 BF16 autocast를 사용한다.

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache"
export TORCHINDUCTOR_COMPILE_THREADS=2
mkdir -p results
(
  exec 9>/tmp/va-gpu.lock
  flock -w 2400 9 || exit 1
  probe=h200/wave22a_gemm_probe
  python "$probe/step.py" --output results/default1.json --trace results/default.trace.json
  python "$probe/step.py" --arm merge --output results/merge1.json
  python "$probe/step.py" --arm merge --output results/merge2.json
  python "$probe/step.py" --output results/default2.json
  python "$probe/step.py" --cudnn-benchmark --output results/default_search.json
  python "$probe/step.py" --arm merge --cudnn-benchmark --output results/merge_search.json
  python "$probe/step.py" --arm reduce-overhead --output results/reduce_overhead.json
  python "$probe/step.py" --arm graph --output results/graph.json
  python "$probe/step.py" --arm tuned --output results/tuned.json
  python "$probe/conv.py" --output results/conv.json
)
python h200/wave22a_gemm_probe/audit.py \
  --trace results/default.trace.json --output results/audit.json
```

- `step.py`는 첫 스텝 비용, warmup 후 wall 및 CUDA 이벤트 시간, 피크 할당 및 예약 바이트, 유한 loss/gradient를 기록한다. 옵션별 첫 스텝 시간에는 해당 프로세스에서의 캐시 재사용이 반영된다. CUDA Graph의 수동 캡처 시간은 별도 기록한다.
- `conv.py`는 첫 번째와 두 번째 stem의 fprop, 입력 adjoint, weight adjoint를 별도의 BF16 연산으로 측정한다. 실제 학습은 입력 이미지의 adjoint를 요청하지 않는다.
- `audit.py --fx FORWARD BACKWARD`는 해당 트레이스와 같은 variant 및 배치의 `fx_graph_readable.py` 두 개를 받는다. 연산 순서가 맞을 때 FX tensor shape과 소스 위치를 연결한다. 같은 디렉터리에 `output_code.py`가 있으면 shape가 일치하는 명시적 stride도 읽는다.
- 학습 프로파일의 split-K 런치들은 ATen 호출별로 묶어서 계산량을 한 번만 센다. 논리적 I/O만으로 계산한 GB/s에는 중간 버퍼 트래픽, cache hit 또는 실제 DRAM 바이트가 반영되지 않는다. FP32 행의 Tensor Core 기준 시간은 BF16 피크를 가정한 참고값이다.
