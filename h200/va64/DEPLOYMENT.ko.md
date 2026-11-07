# VA64 G H200 실행

이 저장소는 VA64 terminal energy readout을 signed complex-feature GAP로 교체한 G를 담는다. 기존 4090 B/P/A 실행을 변경하거나 중복 실행하지 않는다.

## 첫 요청: preflight

`bash h200/va64/run.sh preflight`

Python3.13.11과고정된패키지환경을별도로만들고,wholeH200/데이터manifest/CPU·GPUgradient검사/224px20step/새프로세스checkpoint복원을검증한뒤종료한다. **100epoch학습은시작하지않는다.** `/app/output/va64-g-seed0-preflight-20261107`에로그·환경·데이터manifest·smoke checkpoint·복원검사결과를남긴다. 기존폴더가있으면덮어쓰지않고실패한다. 재시도는기존job종료/출력을확인한뒤새VA64_RUN_ID를명시해야한다.

데이터기본후보는`/app/data/ImageNet-2012`이다. 현재존재가확인된경로가아니며스크립트가직접검증한다. 파일목록/크기해시는현재4090 ImageNet과일치해야한다. stdout에는전체history나비밀정보를출력하지않는다.

## 본학습은 별도 요청

H200 preflight와checkpoint회수·원격중단경로를실제확인한후별도run ID로실행한다.

`VA64_OPERATIONS_VERIFIED=yes VA64_RUN_ID=va64-g-seed0-train-UNIQUE bash h200/va64/run.sh train`

위환경변수는외부백업/중단을구현하는기능이아니라운영확인을기록하는gate다. 확인전임의로yes로설정하지않는다. 본학습코드는G/seed0/100epoch、physical128·accum2·GPU Mixup·BF16이다. 공통backbone은initialization/initial-B0.pt의검증된미학습B0tensor에서복원하고Ghead는별도초기화한다. 새source/config계약으로저장하며기존B/P/A checkpoint를resume하지않는다.

정상중단은runner의STOP파일을epoch경계에서확인한다. 컨테이너외부에서STOP을만드는실제접근경로는아직확인하지못했다. checkpoint저장위치는플랫폼관리자가회수할수있는/app/output이며외부장기백업성공은아직주장하지않는다.

## G 정의와 해석

네방향terminalscan/gain정규화/reader/damping/phase를유지한다. 각복소state의Re/Im을FP32공간평균하고direction→component→pole순서로concat한다.640차원BN(affine=False)+Linear1000이며총파라미터1882760개다. B의320차원head보다320000개크므로동일head capacity라고해석하지않는다. G0는한seed이며반복seed유의성/보편적인energy우월성을주장하지않는다.

학습소스는SOURCE_MANIFEST.json,추가진입점/config/초기값은DEPLOYMENT_MANIFEST.json으로고정한다. 실제요청은git commit SHA를checkout하여실행해야한다.

## 저장소 접근

저장소는기본비공개다. H200인프라의clone권한은이preflight요청전별도로확인해야한다. 브라우저에서저장소가보이는것만으로H200 clone가능성을보장하지않는다. 토큰을이슈본문이나명령에붙이지않는다.
