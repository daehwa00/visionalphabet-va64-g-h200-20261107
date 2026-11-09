# APPLE70M H200 스트리밍 사전검사

이 진입점은 데이터 접근성 검사용이다. 전체 데이터 다운로드나 본학습은 시작하지 않는다.
사용자 확정 목표: APPLE70M scratch, 약100B tokens, 기존 논문 recipe 기준.

- 임시 tokenizer/부분 token cache: `/app/scratch/<run>/`.
- 영구 결과·환경·검증 report·console: `/app/output/<run>/`.
- 공식 Pile의 immutable revision·shard metadata/hash를 고정하고, 첫 shard의 처음/중간/끝을 각각8MiB만 읽는다. 서버가 Range를 무시하면 body를 읽지 않고 실패한다.
- W&B는 로컬 observer가 `daehwa/apple70m-h200`에 기록. API key를 H200/공개코드/이슈에 전달하지 않는다.
- 전용 Cloudflare Worker로 기존 VA64 작업과 중단 상태를 분리한다. owner `stop`은 자식 프로세스 그룹을 종료하며 checkpoint/백업/검증을 수행하지 않는다. 기본 최대 grace30초 후 강제종료, 통신 정상 시 polling5초+HTTP지연이 추가된다.
- W&B 준비 확인 전 다운로드하지 않는다. 사전검사는 제어 연결/observer가 장시간 끊기면 중단한다.

## 미해결 본학습 조건

기존 NeoX sampler는 preshuffled corpus에서 다시 document/sample shuffle을 한다. 단순 순차 stream으로 바꾸면 데이터 순서가 다르므로, 작은 LRU/range cache로 동일 접근 순서를 재현할 수 있는지 검증해야 한다. 부분다운로드 성공만으로 충분한 처리량이나 동일 순서를 주장하지 않는다.

기존 HF 속도 runner는 pretrained BF16이고 기존 paper runner는 scratch FP16/DeepSpeed다. 현재 fast-loss70M 결과를 논문조건 scratch로 그대로 이식하지 않는다. 초기화/precision/optimizer decay/clip/sampler parity 후 H200 microbatch tuning이 필요하다.

보수적100B 정의: S2048/global1024 유지, 원래143000-update cosine schedule 첫47684 updates(100,000,595,968 tokens), warmup1430 유지. 원래 validation prefix 중복 위험이 있어 별도 공식 validation을 준비해야 한다.

## 로컬 운영

`python h200/apple/observer.py --root "$HOME/experiments/apple70m-h200-observer-20261109" --secrets "$HOME/.config/visionalphabet/apple70m-h200-20261109/owner-secrets.json"`

같은 명령에 `--action status`, `--action stop` 또는 필요한 경우에만 `--action force`를 붙인다. 이미 실행 중인 H200 job과 중복 제출하지 않는다. 이슈 제출은 사용자가 한다.
