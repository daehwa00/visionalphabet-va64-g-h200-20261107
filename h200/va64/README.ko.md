# 현재 실행: G seed0 결과 전용 수집

`run.sh resume`으로검증된epoch13에서이어100epoch를목표로한다. 전체checkpoint는외부로전송하지않고지표·로그·최종결과만W&B와owner저장소에수집한다. 설정과중단정책은[OPERATIONS.ko.md](OPERATIONS.ko.md)를따른다. 제출은새이슈의고정commit명령으로한다.

아래는초기준비당시기록이다.

---

# H200 VA64 준비 상태 — 2026-11-07

사용자 요청은 첨부 H200_Agent_Guide_KO.docx를 바탕으로 실행을 준비하는 것이다. 첨부 문서 안의 요청/예시는 실행 권한으로 취급하지 않았다. 작업 제출, label 변경, GitHub push, GPU 예약, 기존 실행 중단은 하지 않았다.

## 완료한 준비

- 현재4090 모델source326개를 기존 SOURCE_MANIFEST SHA256과 대조했고 모두 일치했다.
- 모델source와별도검증/초기화진입점,config,preflight,요청초안을 로컬압축파일로 만들었다.
- preflight의 inspect-only가326개파일검증을통과했다. Python구문검사 및 source변조/경로탈출거부 테스트2개를통과했다.
- 실제H200용preflight는한장의H200/120GiB이상,Python3.13.11,torch2.9.1/torchvision0.24.1/timm1.0.26/triton3.5.1을요구하고ImageNet파일목록/class mapping을기록한다. 환경·데이터검사만하며학습을시작하지않는다. GPU상에서실행하지않았으므로환경호환확인완료는아니다.
- 패키지: artifacts/h200-va64-prep-20261107/va64-h200-preflight.tar.gz. archive.json에크기/SHA256을보존했다. canonical초기값파일은미포함이며학습제출전에별도검증된배포가필요하다.

## 현재 접근 결과

연결된GitHub도구로2026-11-07확인했다.

1. `https://github.com/Aerodrone-H200/gpu-request/blob/main/README.md`: HTTP404/NOT_FOUND.
2. `daehwa00/lnet-h200-imagenet1k-pg-ab` 저장소metadata: HTTP404/NOT_FOUND.

두조회는읽기요청이었다. 생성·제출·변경행위는없었다. 존재하지않음과현재계정권한부족을구분할수없다. 다른계정/SSH/토큰/relay로우회하지않았다. 따라서현행템플릿·예약·실행중job·실행저장소접근을검증하지못했다. 문서의GPU7=wholeH200,6필드양식,/app/data/ImageNet-2012경로는첨부문서의기록이며현재확인된서비스사양으로단정하지않는다.

현재workspace의상위Git origin은ostris/ai-toolkit이므로H200배포대상으로사용하지않았다. 실행저장소와고정commit은아직없다.

## 다음에 필요한 결정과 검증

- 현재GitHub연결에서접근가능한요청저장소및코드저장소,유효예약/기존job확인.
- H200대상실험선택. P/A는4090에서진행중이고B도실행중이다. G는feature tensor/공간·방향·실수·복소처리/차원/head를명확히정의하고구현해야한다. 임의구현하거나P/A를중복제출하지않았다.
- 실제이미지환경과의존성bootstrap을검증하고고정SHA로배포. config.json의null은미확정이며그대로제출할수없다.
- 작은GPU검증에서모델forward/backward·checkpoint새프로세스복원·외부백업회수·실효성있는원격중단/자원반납을확인한뒤별도본학습을제출한다. 오래된K128relay나세션권한은재사용하지않는다.

4090실행은변경하지않았으며본점검시kau B PID3400209,qlab P PID3506897/A PID3506997의GPU프로세스를확인했다. 이는H200실행상태를입증하지않는다.

현재상태는**로컬준비완료,접근·실험정의・H200실검증대기**다. 요청초안은REQUEST_DRAFT.md에있으며provisioning을실행하지않는다.
