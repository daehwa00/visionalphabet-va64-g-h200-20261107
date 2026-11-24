# Wave 25C stem probe

Wave 21C의 Stage 0 Conv3×3 stride 2, 채널 LayerNorm, exact GELU를 비교한다.
`LNET_FUSED_STEM_CONV=off`는 Wave 21C의 경로를 실행한다.
`auto`와 `on`은 기존 fused stem 진입 조건과 새 커널의 지원 조건을 만족할 때 CUDA FP16/BF16 커널을 실행한다.
입력 gradient가 필요한 경우 기존 경로로 돌아간다.

GPU를 숨긴 환경에서 오프라인 컴파일 및 리소스 확인을 실행한다.

```sh
export CUDA_VISIBLE_DEVICES=""
PYTHONPATH=src python h200/wave25c_stem_probe/offline.py > offline.log
python h200/wave25c_stem_probe/resources.py offline.log
```

gunsoo의 격리 스테이징 디렉터리에서 CUDA Graph와 회귀 검사를 실행한다.

```sh
(
    flock -w 2400 9 || exit
    PYTHONPATH=src:scripts python -u h200/wave25c_stem_probe/isolated.py
) 9>/tmp/va-gpu.lock
bash h200/wave25c_stem_probe/verify.sh /path/to/python
```

확인 순서는 각 cuDNN benchmark 모드의 지정된 A/B 순서와 동일하다.
`verify.sh`의 두 번째 인자로 `training` 또는 `regression`을 전달하면 해당 검사만 실행한다.
프로파일의 `self_cuda_ms_by_bucket`은 CUPTI가 기록한 개별 CUDA 커널 시간의 합이다.
`cudnn_conv`는 convolution과 NHWC 채널 패딩을, `fused_stem`은 두 스테이지의 native stem 커널을 포함한다.

Tensor Core 비교는 `WAVE25C_TENSOR=1 WAVE25C_MAIN_ONLY=1`을 지정해 같은 CUDA Graph
프로브에서 실행한다. 실험용 backward의 타일 행 수, 워프 수, 타일 합산, CTA 수는 각각
`WAVE25C_TC_M`, `WAVE25C_TC_WARPS`, `WAVE25C_TC_TILE_REDUCE`, `WAVE25C_TC_CTA`로 바꿀 수 있다.

검증 스크립트는 cuDNN benchmark 쌍과 controlled ABAB 각각의 실행 중 GPU 락을 유지한다.
중단한 실행을 이어갈 때 `WAVE25C_RESUME=1`을 지정하면 종료 코드 0이 기록된 arm을 건너뛴다.
