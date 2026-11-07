# 제안: 리버탈용 G — terminal complex feature GAP

상태: 구현 및 CPU/RTX4090 검증 완료. 224px compiled BF16 20-step·새 프로세스 checkpoint 복원 통과. H200 실행 및 100 epoch 학습은 아직 미시작이다.

## 직접 답할 리뷰 질문

첨부 OpenReview의 tmcT는 frozen-checkpoint intervention만으로 재학습한 phase-free/real recurrence/conventional feature pooling 대비 기여를 보일 수 없다고 지적했다. G는 conventional feature-pooling readout을 처음부터 재학습하는 비교를 목표로 한다. P는 phase0을검증하지만real-valued recurrence를검증하지않는다. A는terminal directional energy 분리의 실용성을검증하며전체backbone의방향scan필요성을검증하지않는다.

## G의 권장 정의

VA64의 stem, 모든 same-resolution block, reader, damping/phase, 네 방향 terminal scan 및 gain normalization을 B와 동일하게 유지한다. terminal scan이 생성하는 gain-normalized 복소 상태를 Z_d[b,h,w,p]라고 한다.

B descriptor: 각방향·pole의 log1p(mean_hw(abs(Z_d)^2)), shape[B,4P]=[B,320].
G descriptor: 각방향·pole의 mean_hw(Re Z_d)와 mean_hw(Im Z_d)를 모두 이어붙인 값, shape[B,8P]=[B,640]. 평균은FP32에서계산한다. G에는abs²/log1p/energy평균을넣지않는다. 네방향평균을먼저내거나terminal reader/scan을제거하지않는다. 방향과real/imag의순서를config와테스트에고정한다.

G classifier: 기존과같은 BatchNorm1d(affine=False)+Linear(1000). 입력차원만640으로바뀐다. B affine head는321000개, G는641000개파라미터가예상된다. 다른학습파라미터가정확히같다면G총계예상은1882760개다. 이는실측값이아니며구현후inventory로확인해야한다.

G의head가더크므로headcapacity가같다고쓰지않는다. B가G보다좋아도‘이번설정에서energy readout이일반GAP대안보다유리했다’까지만해석한다. 모든pooling/head대안에대한우월성,energy의필수성,순수한통계량효과를주장하지않는다. 반대결과도그대로보고한다.

## 피해야 할 구현

- terminal입력feature만GAP하면서terminal reader/scan을삭제하면encoder/compute까지바뀐다. 별도실험으로는가능하지만이번readout-only G와혼동하지않는다.
- real/imag를버리고실수부만선택하거나학습결과를본뒤pooling정의를바꾸지않는다.
- CPUreference는가능하지만GPU에서full-spatial states를매번materialize하는naive구현을최적화된B와직접latency비교하지않는다. 우선수학/gradient검증후공정한runtime측정을별도로한다.

## 실행 전 검증

1. 동일seed의공통backbone초기tensor가B와동일하고정의된head차이만있는지검사한다.
2. CPUreference와GPU/BF16출력·gradient를비교하고zero/constantfield및공간평균의signed cancellation을검사한다.
3. 네방향state/axis순서,FP32평균,phase/damping/reader유지와정확한param수를확인한다.
4. checkpoint새프로세스복원·optimizer/RNG연속성과mode/source불일치거부를확인한다.
5. H200GPU/데이터mount/manifest,작은실데이터학습,외부checkpoint회수및원격중단경로를검증한다.

## 비교와 자원 우선순위

현재진행중인新B0/P0/A0과같은seed0·100epoch·데이터전체membership·recipe·finalepoch선택으로G0를H200에배정하는것을우선권장한다. physical128/accum2/global256과GPU Mixup을유지한다. H200속도를4090속도와직접비교하지않는다.

과거B seed501/509/521은참고baseline으로보존하되새seed0과paired차이라고부르지않는다. 기존B는drop_last=True,현재는전체마지막partialbatch까지사용하므로정확히동일recipe라고쓰지않는다. 이미실행중인新B0를기준으로쓰면이차이를줄일수있다.

첫B/P/A/G0이끝나면단일seed탐색결과로보고한다. 추가budget이확보될경우동일한B/P/A/G를seed1·2에서도완성하여군별mean±sampleSD와seed내차이를보고하는안을권장한다. 자동추가실행하지않는다. 실용차이0.3pp는사전기록값으로유지하되seed1개의작은차이로동등성/유의성을주장하지않는다.

리뷰의compute/retention지적은이ablation만으로해결되지않는다. 다음우선순위는공식RMT또는명확히정의한retention baseline과같은학습protocol·인접/맞춘계산budget비교다. 새baseline을추가하기전기존측정의FLOPs/latency/params를함께명시하고parameter compactness를compute efficiency로바꾸어말하지않는다.
