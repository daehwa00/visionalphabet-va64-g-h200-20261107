# 데이터 접근 실패 2개 항목만 재평가

첫 추가 평가에서 20개 model-family 조합 중 18개가 성공했다. residual-final의 xStoryCloze는 HF 파일 조회 실패, residual-best의 BLiMP는 HF API 429로 실패했다. 성공한 조합은 다시 실행하지 않는다.

고정된 원본 revision의 Parquet 89개와 동일 tokenizer를 사전 다운로드해 파일 SHA256·바이트 수·행 수를 검증한다. H200에서는 검증된 ZIP을 내려받고 다시 검증한 뒤 `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`로 평가한다. 데이터 내용·프롬프트·split·샘플 수·0-shot·모델 가중치·FP32 설정을 유지하고 데이터 로딩 경로만 로컬 파일로 바꾼다.

`evaluate.py --families ... --dataset-root ...`로 요청한 한 family만 실행하며 summary의 requested_families에 범위를 기록한다. 후속 병합 시 두 실패 항목만 보충하고 원래 실패 기록과 retry protocol을 보존한다.
