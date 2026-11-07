# 현재 운영 정책: 결과만 외부 보존

- G seed0,100epoch,physical128×accumulation2(global256),기존recipe유지.
- `bash h200/va64/run.sh resume`: 검증된epoch13/step65065를시작시에한번다운로드해epoch14부터계속한다. 속도비교는이미완료했으므로반복하지않는다.
- 학습중과종료시전체checkpoint의외부전송·검증·ACK대기를하지않는다. latest/best는컨테이너로컬에만남으며컨테이너삭제시사라질수있다.
- 지표·epoch곡선·최종정확도/loss·시간·메모리·로그만외부수집기로보낸다. W&B장애나수집기/relay연결오류는학습을중단시키지않는다. 원격kill switch의명시적stop/force와실제학습stall감시는유지한다.
- 최종요약은W&B지표/작은결과artifact,owner의JSON/JSONL에보존한다. 연결장애가끝까지계속되면W&B전달을보장할수없으므로 `VA64_FINAL_SUMMARY=` 표식으로H200서비스결과로그에도출력한다.
- 중단요청시validation·새checkpoint·외부백업없이optimizer경계에서종료한다.
- 아래내용은이전checkpoint백업정책의운영이력이다. 현재학습경로에는적용하지않는다.

---

# G 원격 운영: Cloudflare + 외부 W&B observer

## 데이터 경로

H200 guard → G 전용 Cloudflare Worker → 외부 observer의 디스크 + W&B.

H200에W&B키를배포하지않는다. guard는매실행새agenttoken을메모리에서생성하고,등록된H200송신IP·허용commit·pod형식검증을거쳐session을받는다. owner권한은H200에전달하지않는다. 외부observer가자신의기존W&B인증으로`daehwa/visionalphabet-rebuttal-va64`비공개프로젝트에기록한다. 공개저장소나이슈에비밀키를넣지않는다.

로그/지표는guard가15초간격으로전송하고observer가외부디스크와W&B에기록한다. 전체종료로그는W&B logs artifact로보존한다. 모델checkpoint는각완료epoch와preflight종료때Cloudflare로업로드한다. observer는SHA256·압축파일안전성·G모델strictload·optimizer/RNG존재를검사한후W&B model artifact업로드완료를기다린다. 그후Cloudflare에ACK한다. H200은이외부확인전다음epoch로진행하지않는다. 실패시마지막checkpoint와업로드진행기록을보존하고실패하며학습을조용히계속하지않는다.

W&B메트릭과checkpoint백업은별개다. metrics만보인다고백업성공으로판정하지않는다. preflight는synthetic20step이고본학습정확도결과와분리한다. ownerobserver는H200컨테이너와별도인상시호스트에서계속실행해야한다.

## 중단

owner호스트에서다음을실행한다. 비밀파일은공개저장소밖0600권한으로보관한다.

```bash
PYTHONPATH=src:scripts python scripts/va64_h200_control.py status --secrets ~/.config/visionalphabet/va64-g-20261107/owner-secrets.json
PYTHONPATH=src:scripts python scripts/va64_h200_control.py stop --secrets ~/.config/visionalphabet/va64-g-20261107/owner-secrets.json
```

stop은다음controlpoll에서ROOT/STOP을기록한다. 본학습은현재epoch의validation/checkpoint및외부백업을완료한후멈춘다. 따라서즉시종료가아니며실제응답시간은남은epoch에따라달라진다. guard의기본grace는20분이며그이후또는명시적force요청시해당G자식process group만강제종료한다. 강제종료는미완료epoch를잃을수있고,마지막외부확인checkpoint를기준으로복구해야한다. preflight는step경계에서STOP을확인하며compile진행중에는반응이지연될수있다.

observer/relay연락이10분이상없으면guard가중단을요청한다. 완료/중단요청/실제종료/백업완료를각각따로기록한다. 새worker의`/va64g`scope만사용하며기존K128작업이나relay를변경하지않는다.

## 배포와 실행

`h200/va64/cloudflare/wrangler.jsonc`는새worker `visionalphabet-va64-g-20261107`과새Durable Objects를정의한다. `h200/va64/deploy.sh`는현재gitcommit을허용하고,owner secret은stdin으로만주입한다. H200송신IP기본값203.253.146.151은이전운영기록이며새preflight에서일치해야한다. 변경된IP는계속거부하며검증없이allowlist를넓히지않는다.

1. 유효한Cloudflare배포로그인을확인하고새전용owner secret을배포한다.
2. 외부호스트에서`bash h200/va64/owner.sh`를tmux등지속프로세스로실행한다.
3. 실제worker health/auth와owner heartbeat/W&B연결을확인한뒤새commit이채워진H200이슈를사용자가제출한다.
4. H200실제자원에서중단·checkpoint회수까지확인한후별도의100epoch본학습을제출한다.

기존`run.sh`는guard없이실행하지않는다. observer가W&Bready를확인해주기전환경준비/학습을진행하지않는다. 명시적인W&B연결실패시disabled로자동대체하지않는다.

## 현재 확인한 것과 남은 것

- W&B실제로그인사용자는daehwa이고프로젝트access=PRIVATE를확인했다.
- 로컬Cloudflare실행환경→실제W&B에19,874,323byte checkpointbundle을업로드했다. 다시다운로드한SHA256이1911ba5a15c8c4bdcc09d424f6025a14bda9f3c2daa801177e814af072f0b195로일치했고G strictload를통과했다.
- 로컬실제guard/자식프로세스에ownerstop을전달해강제종료없이exit130과기존checkpoint보존을확인했다. 실제H200에서의중단시험은아니다.
- 전송/ACK순서/비밀파일제외/epochcallback검사22개를통과했다.
- 사용자OAuth재로그인후G전용worker를실배포했고owner인증/health/상시observer heartbeat를확인했다. 라이브테스트영역에서동일checkpoint의전송·W&B확인ACK와실제guard중단을검증했다. 이는owner권한으로격리된테스트영역에서수행한검사이며실제H200pod/송신IP/agent등록검증은첫H200preflight에서수행한다.
- 라이브회수에서자동Brotli응답이ContentDecodingError/ChunkedEncodingError를발생시켜Accept-Encoding: identity로고정했다. 같은256KiB조각이비압축응답에서는정상수신됐고전체19,874,323bytebundle의해시가일치했다. 실패시도는operations/live,수정후성공은operations/live-v2에별도보존했다.

H200에제출할수있는것은먼저20step preflight다. 실제H200에서등록·데이터·학습·중단·외부회수까지확인한후100epoch본학습으로진행한다. 저장소최신commit을허용하는worker재배포와observer동작을제출직전에대조한다.


## 2026-11-07 재시도
첫 H200 요청 #663은 uv의 CUDA 인덱스 우선 탐색으로 requests 설치 전에 실패했고 자원이 반납됐다. 패키지 버전은 유지하고 torch/torchvision만 공식 CUDA wheel URL과 SHA256으로 지정한다. 나머지는 PyPI에서 설치한다. 외부 observer는 종료된 세션 저장 후 다음 세션을 기다리며 재시도별 상태를 분리한다. 새 요청은 별도 RUN_ID attempt2로 실행한다.


## Batch 비교 후 epoch 경계 resume 및 즉시 중단
`run.sh tune-resume`은 외부 검증된 부모 checkpoint의 archive/hash/계약을 고정해 복구하고 실제 ImageNet으로 128×2와256×1을 각각 별도 프로세스에서 측정한다. Compile warmup30 update 뒤100 update의 전처리·전송·GPU Mixup·학습 전체 처리량과peak memory로 선택한다. 256 OOM이면128을 사용하며, 다른 실패는 중단한다. 선택 결과는 telemetry/batch-comparison.json으로 다음 checkpoint와 함께 외부 보존된다. 실제 H200 비교는 이슈 실행 시 수행한다.

정상 resume 계약은 엄격하게 유지한다. 이번 전환만 명시적 resume-plan으로 원본 hash와 계약, 허용된 대상 소스 hash,동일 recipe를 확인한 뒤 모델·optimizer·RNG·완료 epoch·global_step을 보존한다. global256,LR,AdamW,스케줄,증강 설정은 유지한다. Epoch 경계 sampler 순서는 유지하지만 worker의 증강 RNG, Mixup 짝/난수,BN 통계와 마지막143장 가중치는 physical batch에 따라 달라지므로 수치적으로 동일한 trajectory를 보장하지 않는다. 새 run의 parent lineage와기존 best를 보존한다.

사용자 요청에 따라 새 실행의 STOP/SIGTERM은 현재 optimizer update 경계에서 validation·새 checkpoint·최종 외부 backup 없이 exit130한다. 이미 validation 중이면 다음 validation batch 전에 멈춘다. Backup 중 SIGTERM은 동기 전송/재시도를 취소하며 background upload thread를 기다리지 않는다. 기존 외부 checkpoint와 수집된 로그는 유지한다. 따라서 다음 resume는 마지막으로 외부 저장된 epoch부터이며 중단한 epoch의 미저장 update는 잃는다. 이전 commit으로 이미 실행 중인 job에는 이 변경이 소급 적용되지 않는다.


### 기존 #665 중단과 복구 결과
기존327a9a2 guard가20분 grace_expired로자동강제종료했다. 수동force명령은없었다. 자원반납확인. 원본archive는160chunk중146개만전송됐지만 완전한epoch1 best.pt를복구했다. ZIPCRC검사,원래계약의hash재현/일치,원래datasetmanifesthash일치,G strict모델과AdamW load를통과했다. 복구모델/optimizer/RNG/history를보존한정규화archive는W&B와relay에완전저장됐고 W&B 독립다운로드SHA도일치했다. 원본부분전송본과복구증거를private W&B recovery artifact로보존했다. resume-plan은이복구archive를명시적으로고정한다.
