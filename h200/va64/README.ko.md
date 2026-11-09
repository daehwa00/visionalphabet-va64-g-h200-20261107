# H200 G seed 1·2 실행

`bash h200/va64/run.sh seeds12`으로한H200에서seed1→2를각100epoch처음부터학습한다. 128×2(global256),기존recipe와seed별P/A공통초기화를유지한다. 실패·중단이면다음seed를시작하지않는다. 모델checkpoint는컨테이너로컬에만저장하고지표·로그·최종결과만외부보존한다. 제출은고정commit의새이슈폼으로한다. 상세는[OPERATIONS.ko.md](OPERATIONS.ko.md).
