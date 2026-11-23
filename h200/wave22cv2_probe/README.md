# Wave 22C-v2 · FP32 PS14 검증

`LNET_CAPTURE_SAFE_ORTHOGONAL=on`은 ablation 빌드 시 PS14를 설치한다.
`off`는 빌드와 학습 준비에서 교체를 건너뛴다. 기본값 `auto`는 빌드에서는
native를 유지하고 기존 학습 준비 훅에서 PS14를 설치한다. 토글은 모델 생성 전에 설정한다.

`PYTHONPATH=src:scripts`를 지정하고 저장소 루트에서 실행한다.

- `matrix.py --device cpu --coordinates <trained-coordinates.pt> --output <json>`은 순전파,
  벡터–야코비안 곱, Gram 오차를 계측한다. 같은 좌표 파일로 `--device cuda`를 실행한다.
- `step.py --arm <native|safe|reduce-overhead> --output <json>`은 별도 프로세스에서
  준비 4회와 계측 8회를 실행한다. 각 계측 순서는 native → safe → reduce-overhead,
  reduce-overhead → safe → native다. 두 순서를 각각 `--benchmark` 유무로 반복한다.
  `--trace <json>`은 계측 후 한 스텝의 CUDA 런타임을 기록한다.
  `--arm fp64 --warmup 1 --count 1`은 행렬 지수만 FP64로 계산하는 비교군이다.
- `gates.py --gate dynamo --output <json>`과
  `gates.py --gate fold --variant <model> --output <json>`은 기존 H200 검증 코드를 사용한다.

학습 비교군은 모두 BF16 autocast, seed 501, FP32 matmul precision `highest`를 사용한다.
GPU 실행은 공유 GPU 잠금을 획득한 셸에서 실행한다.
