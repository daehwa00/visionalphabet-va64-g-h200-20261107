# APPLE70M: checkpoint + benchmark 실행

이 코드는 기존 가중치 미저장 실행을 대체하는 새 scratch 실행이다. 이전 실행의 가중치는 이어받지 않는다. V-only H200과 V-AV kau는 같은 seed1234/global1024/S2048/100B 목표 및 순차 Pile 조건을 사용한다. 이전 논문 데이터 shuffle 순서와는 다르다.

## 저장

- H200 결과 root: `/app/output/<run-id>/`.
- 실제 학습 checkpoint: `train/checkpoints/latest.pt`, `previous.pt`.
- validation best와 최종 평가용 모델: `train/checkpoints/best.pt`, `final.pt`.
- 각 `.pt` 옆의 `.pt.json`에 SHA256/크기/step을 기록한다. **파일과 같은 이름의 sidecar를 함께 회수한다.** previous의 receipt.path는 원래 발행된 latest 위치일 수 있으므로 파일 이름 쌍을 기준으로 받는다.
- 첫 optimizer step, 이후500 globalsteps(약1.05B tokens), 마지막에 재개용checkpoint를 저장한다. latest/previous 두 세대만 보존한다.
- 재개용파일에는 model/optimizer/scaler/RNG/token cursor/skip count와 과거best 가중치가 들어간다. 복원 시 source/variant/recipe/data 계약이 다르면 거부한다.
- 임시파일+파일fsync+atomic rename+디렉터리fsync. manifest도fsync한다. 중단명령에는추가checkpoint/평가/백업을 수행하지 않는다.
- **W&B에는 지표/로그/저장파일 메타데이터만 기록한다. 가중치 업로드는 하지 않는다.** 관리자가 `/app/output`에서 파일을 꺼내주는 회수방식은사용자확인사항이다.

## H200 단계

환경설치→CPU/GPU tests→독립validation준비→microbatch8/16/32/64/128 측정→선택batch에서저장/새프로세스복원검증→8tasks각2문항GPUsmoke→실데이터20step→새로초기화해47684step 본학습→final/best모델8종전체평가.

모든 사전검증을 통과해야 본학습을 시작한다. fullbenchmark 실패가 있어도 가중치는 `/app/output`에 남는다. 결과는 training 완료와 benchmark 완료를 구분한다.

## 평가

LAMBADA OpenAI, PIQA, WinoGrande, WSC, ARC Easy, ARC Challenge, SciQ, LogiQA. 0-shot, 정식평가limit없음, 양변형동일 FP32 inference/max_length2048. LogiQA는공식parquet변환commit을고정하며upstreamprompt/label은그대로다. 각task문항수/metric유효성검사,누락/실패는완료로표시하지않는다. 소량검사는formal_complete=false다.

`benchmarks/final/summary.json`, `benchmarks/best/summary.json`과 task별JSON을회수한다. 코드commit, source/data manifests, resolved-config, train/validation.jsonl, steps.jsonl도함께보관한다.

## 복원

파일을 같은원본코드/환경으로복사한뒤, **새 output 경로**로 train.py에 `--resume /path/latest.pt`를 전달한다. sidecar필수다. 데이터manifest내용/globalbatch/microbatch/목표steps/variant는기존계약과같아야한다. 데이터/validation파일의절대경로는달라도내용해시는같아야한다. latest손상시검증된previous를사용한다. 가중치만있는final/best는optimizerresume용이아니다.

## 검증 근거와 제한

70M GPU에서모델/optimizer/scaler/RNG복원직후비트일치,새프로세스이어학습성공. 독립CUDA실행과비교시모델상대L2약4.6e-8의부동소수점차이를관측했다. 양변형각8task×2문항GPUsmoke통과. historical-best보존/sidecarfsync수정후residual GPU검증및8tasksmoke재통과. 약1.1GB fullresumecheckpoint를kau밖으로실제복사해SHA검증했다.

H200새코드는아직제출전이며H200자체검증은첫실행gate에서수행한다. `/app/output`관리자회수는실제로관리자가파일을제공한뒤해시검증해야완료된다. 관리자회수없이직접다운로드가된다고주장하지않는다.
