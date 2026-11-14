# APPLE V−AV 4090 → H200 재개

27,000 step / 56,623,104,000 tokens 체크포인트를 비공개 GitHub release에서 받아 SHA256 검증 후 복원한다. 임시 다운로드 주소는 인증된 relay로만 전달하며 GitHub 소유자 토큰은 H200에 전달하지 않는다.

microbatch 8 × accumulation 128 = global batch 1024의 원래 설정을 유지한다. H200에서 microbatch 64는 2-step 비교 relL2 2.6287e-4로 1e-4 기준을 넘었으므로 채택하지 않는다. 배치 8의 H200 정상 step은 약 323,804 tokens/s로 확인됐다.

전송 SHA 검증 → 원래 계약 및 GPU model/optimizer/scaler/RNG 복원 해시 검증 → 실제 2-step 학습·저장 → 새 프로세스 복원 감사 → 8-step 속도 검증(중앙값 > 220k tokens/s) → 본학습 → final/best 8-task 정식 평가 순서다. 원본 best/cursor/optimizer 상태를 유지하며 GPU 산술 차이로 이후 궤적의 bitwise 동일성을 보장하지 않는다.

H200 본학습 확인 전까지 qlab GPU 1 학습을 유지한다. qlab보다 잔여시간이 짧을 때만 전환한다. 모든 원본 출력은 보존한다. STOP은 추가 저장/평가 없이 종료한다.
