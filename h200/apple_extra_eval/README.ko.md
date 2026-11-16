# APPLE 추가 5개 벤치마크

검증된 V-only/V−AV final/best 4개 weights만 평가한다. 재학습 및 기존 8개 과제 재평가는 하지 않는다. 전체 split, 0-shot, lm-eval 0.4.13, FP32, batch 16, GPT-NeoX tokenizer SHA 및 dataset revision을 고정한다. 각 모델 시작 시 원본 source contract와 model state strict load/SHA256을 검사한다.

- BLiMP: 67개 하위 과제 정확도 및 균등 평균.
- SWAG/HellaSwag: 일반 및 정규화 정확도.
- xStoryCloze: 영어 단독, 11개 언어별, 11개 언어 균등 평균을 구분한다.
- WikiText-2: `wikitext-2-raw-v1`, harness word PPL 및 동일 문서별 rolling loglikelihood의 GPT-NeoX token PPL. 문서 연결 sliding-window PPL과 혼동하지 않는다.

최초 모델에서 2-sample diagnostic을 실행하고, full evaluation의 모든 task에 original/effective sample 수가 일치해야 완료로 표시한다. diagnostic은 정식 결과로 표시하지 않는다. 하나라도 실패/누락이면 evaluation_failed다.

가중치는 비공개 release에서 다운로드하고 만료 URL은 인증 relay로만 전달한다. GitHub owner token/Drive 공유 링크는 공개 코드나 H200에 저장하지 않는다. 결과 JSON은 인증 relay에 ZIP으로 자동 회수 가능하게 올리며, stdout에는 compact metrics를 남긴다. 관리자에게 추가 파일 요청 없이 작업을 마칠 수 있도록 구성한다.
