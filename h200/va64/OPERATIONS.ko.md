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
- 전송/ACK순서/비밀파일제외/epochcallback검사21개를통과했다.
- 새Cloudflare worker는아직실배포되지않았다. 기존wrangler의OAuth갱신이HTTP400으로실패했다. 배포로그인갱신이필요하다. 최신CLI확인은현재Node20환경요구조건과맞지않았으며글로벌Node환경은변경하지않았다. 로컬검증은기존4.86.0으로수행했다.

이상태에서H200이슈를제출해운영준비가끝났다고간주하면안된다. Cloudflare실배포와observer상시실행및실제H200검증이남아있다.
