# APPLE V−AV 4090 → H200 재개

27,000 step / 56,623,104,000 tokens 체크포인트를 인증된 relay에서 SHA256 검증 후 복원한다. 기존 V-only 실행과 출력은 보존한다. 훈련 코드와 데이터는 변경하지 않고 microbatch 8→64, accumulation 128→16만 변환하여 global batch 1024를 유지한다. 기존 optimizer/scaler/RNG/best와 cursor를 보존한다.

원본 계약 검증 → 변환 및 상태 해시 검증 → GPU 복원 감사 → 2-step microbatch 8/64 수치 비교(relL2 < 1e-4, tokens/skip 일치) → 저장/새 프로세스 재개 → 8-step 속도 검증(중앙값 > 220k tokens/s) → 본학습 → final/best 8-task 정식 평가 순서다. source/data/recipe mismatch는 허용하지 않는다. GPU/배치 reduction 변경 때문에 bitwise 동일 궤적을 보장하지 않는다.

H200 본학습 확인 전까지 qlab GPU 1 학습을 유지한다. 본학습 전환은 모든 gate 통과를 뜻한다. STOP은 추가 저장/평가 없이 종료한다. 각 실행 출력은 /app/output의 새 디렉터리에 보존한다.

런타임 소유자 자격증명은 저장소/컨테이너에 포함하지 않는다. guard가 발급받은 단기 세션 토큰으로 checkpoint shard를 읽는다. 전송 manifest에는 해시/크기만 기록한다.
