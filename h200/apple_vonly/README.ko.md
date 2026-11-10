# APPLE70M V-only 100B scratch

분기는 `AV + gamma * ((V @ D) @ U)`이다. attention AV 기본 경로를 유지하고 보정항에서만 AV 차감이 없다. 기존 source-control forward_bshd가 canonical V-AV를 호출하던 오류를 override로 수정했다. canonical fused APPLE kernel과 whole-model compile은 사용하지 않는다.

## 실행 순서

1. 전용 kill-switch 등록, 로컬 W&B online readiness 확인.
2. whole H200 확인, scratch에 Python3.11.14 및 고정된 torch2.9.1/cu128·Transformers4.57.3 설치.
3. V-only CPU/GPU 출력·gradient, scratch 초기화·optimizer·학습률, 스트리밍·validation tests.
4. 독립 Pile validation 2,097,153 tokens만 스트리밍 토큰화.
5. microbatch8/16/32/64/128을 별도 프로세스로 측정. 20warmup+100measurement microstep을 완전한 GAS배수로 올림. CUDA OOM만 후보탈락, 다른오류는 중단.
6. 측정된 합성 full-model training throughput이 가장 높은 후보 선택. 실데이터20 global-step smoke로64MiB page경계를 넘겨 입력파이프라인검증. 성공optimizerupdate≥16, V-only U변화, finitevalidation 필수.
7. 모델을 같은seed로 새로 초기화하고47684 global steps 본학습. 검증/benchmark weight는 재사용하지 않는다.

최고속 보장은 이 후보 집합의 실측 범위에 한정된다. GPU수치·OOM·네트워크이상은 H200에서만 최종판정한다. 실패하면100B학습으로 넘어가지 않는다.

## 고정 조건과 차이

- 70,451,200 params,6layers/hidden512/8heads, untied50304vocab, S2048, global1024.
- rank4/gamma1, Dstd.02/U0; seed1234, small_init/wang_init; pretrained weights를 읽지 않는다.
- mainLR.001, branchLR.0001, beta(.9,.95), eps1e-8, baseWD.1(LayerNorm/bias제외), branchWD0, clip1.
- 원본 NeoX143000update schedule,1430warmup,minLR.0001. originalfirstLR0. 100B에서 중단:47,684attemptedsteps =100,000,595,968tokens. FP16overflow로skip된update는 별도집계하며 schedule은성공update만진행. 10연속overflow는중단.
- FP32 parameters/gradients/AdamW moments +FP16autocast/GradScaler4096 growth1000. DeepSpeed의hysteresis2·kernel reduction과 정확히 같지는 않은 nativePyTorch runtime이다.
- 사용자 승인대로 official preshuffled token stream을 순차 소비한다. 이전논문의document/sample shuffle과 데이터순서가 다르며, 따라서 V−AV→V만의 순수한 paired효과로 해석하지 않는다.
- 독립validation은pinned Pile validation prefix이며학습stream을validation으로재사용하지 않는다.

## 저장과 중단

- token cache: `/app/scratch/<run>/train-cache`, reader당2GiB한도/64MiB페이지/다음페이지prefetch. 환경·tokenizer·validation토큰도scratch.
- 결과·resolvedconfig·source/data hash·logs·성능비교: `/app/output/<run>`.
- W&B: `daehwa/apple70m-h200`, 별도localobserver가키를보유. H200에W&B키를전달하지 않는다.
- 모델checkpoint/외부checkpoint전송은비활성(results-only). 중단·실패후학습가중치를resume할수없다. 중단시추가평가·검증·백업을하지않는다.
- owner stop은프로세스그룹SIGTERM, 필요할때30초grace후SIGKILL. 관찰자는종료로그를회수한다. 연결장애중원격명령전달은보장되지않으며, 네트워크연결복구후적용된다.

## 검증된 범위

Pinned kau CPU환경에서30tests통과/1GPUtest skip, full70M FP32 CPU2updates유한loss·U변화확인. Mock remote-stop와observerready/sessionisolation tests통과. H200 FP16/fullS2048 실행은 아직이며 이슈실행첫단계에서검사한다.
