# Wave 27C stem 후보

같은 Whole H200에서 `B=1024`로 `LNET_STEM_W27C=off`, `stage0_opt`, `stage0_v2`, `stage01`을 비교한다. `stage01`과 `stage0_opt`의 차이가 Stage 1 재계산 융합의 효과다. `auto`는 검증 전 기준 경로를 선택한다. `LNET_FUSED_STEM_CONV=off`는 Wave 21C 경로를 선택한다.

```sh
export CUDA_VISIBLE_DEVICES=""
PYTHONPATH=src python h200/wave27c_stem_probe/offline.py
```

gunsoo에서 리포지터리 루트 기준으로 실행한다. 검증 스크립트는 공용 GPU 잠금을 획득한 뒤 CUDA Graph, CUDA pytest, Dynamo 및 Fold 검사를 실행한다.

```sh
bash h200/wave27c_stem_probe/verify.sh ~/miniconda3/envs/alphabet/bin/python
```

Whole H200 번들 안에서 스템의 CUDA Graph 타이밍이 필요한 경우 다음 명령을 사용한다.

```sh
PYTHONPATH=src:scripts:. python h200/wave27c_stem_probe/isolated.py --batch 1024
```
