# 정리 내역과 재현 범위

## 원본과의 관계

원본 압축파일의 12개 파일을 보존 대상으로 선정했다. 공통 runner와 수치 계산 코드는 변경하지 않았다. 각 원본 및 배포 파일의 해시는 `source_manifest.json`에서 비교할 수 있다.

변경한 원본 파일은 두 개다.

1. `mismatch_routing_config.json`: `01_mismatch_activation_export`만 남겼다. 논문 범위 밖 permutation ensemble과 evidence-first 실행 설정을 제거했다.
2. `run_mismatch_pipeline.py`: 학습 CSV manifest가 없으면 `train.json`을 읽도록 입력 경로 fallback을 추가했다. 모델·head 선정·확률 점수·학습 목적함수는 바꾸지 않았다.

`reproduce.py`, 설명 문서, `.gitignore`, `.gitattributes`, 실행 경로 테스트는 공개 정리를 위해 새로 작성한 보조 파일이다. 원본 shell launcher의 GPU 1 고정과 서로 다른 출력 경로 대신 CLI로 입력과 출력 경로를 받는다. 원본의 `run_all_experiments.sh`, VARCO steering·causal sweep·VCD·semantic contrast 전용 코드/설정, notebook checkpoint와 pyc는 배포하지 않는다. 세부 제외 목록은 manifest에 있다.

## 동일 수치 재현에 추가로 필요한 것

- 당시 학습·검증 입력 manifest 및 동일한 데이터 버전/이미지 파일.
- 실제 모델 revision, tokenizer/processor revision, 패키지 버전, GPU/CUDA 정보.
- 확정된 `mismatch_calibration.json`, `retrieval_probe.json`, 선택 head 목록 및 임계값. 이 ZIP에 학습된 artifact는 포함돼 있지 않았다.
- 검색 질의·검색기·순위 생성 코드와 당시 링크 목록. 이 배포본은 이미 검색된 링크 이후 단계만 포함한다.
- 보고된 직접 질의 정책의 임계값 교정, paired AUROC 비교, 최종 논문 표·ROC 그림을 만드는 후처리 코드. 이 ZIP에서 완결된 경로를 확인하지 못했다.
- 테스트 객관식 정답 및 평가 기준. 답변 생성 파일과 최종 평가값은 같은 산출물이 아니다.

새 wrapper는 실행 때 확인 가능한 패키지 버전과 실제 명령을 `outputs/run_metadata/`에 기록한다. 모델 ID에 고정 revision이 없는 원본 코드는 그대로이므로 이것만으로 모델 snapshot 재현이 보장되지는 않는다.

## 해석 시 유의할 설정

- `--l2 1.0`은 제공된 PyTorch L-BFGS 목적함수의 계수다. scikit-learn의 `LogisticRegression(C=1.0)`과 같다고 표기하면 안 된다.
- 주 논문 정책은 `activation_mismatch` + `youden`. 원본 calibration 기본 메타데이터의 combined model/FPR 설정과 구별한다.
- Train CV는 head 선정까지 fold 안에서 다시 수행하는 nested CV가 아니다. 주 결과는 Train에서 정한 특징과 가중치의 별도 Validation 평가다.
- 동일 층 무작위 대조는 `analyze_dimension_matched_controls.py`의 `layer_matched_random_4head` 결과다. 다른 random 그룹의 요약을 섞지 않는다.
- 4-head 난수 생성 전에 single-head 추출이 같은 RNG를 소비한다. wrapper는 원본의 single-head 40회를 유지한다. 이후 projection 0회 설정은 4-head 추출 순서를 바꾸지 않는다.
- mismatch/관련성 질의는 공통 runner의 기본 프롬프트, 검색 답변 생성은 `prompts.py`를 로드하는 원본 동작을 유지했다. 둘을 통일하는 것은 단순 코드 정리를 넘어 실험 조건 변경이다.
- 원본 공통 runner와 분석기에는 논문에 보고하지 않은 기능과 진단 출력이 남아 있다. 파일 의존성을 보존한 배포본이며 최소 메서드 단위 재구현은 아니다.

## 검증 범위

배포 정리 환경에는 PyTorch/Transformers와 실험 GPU·데이터가 없었다. Python 구문 검사, JSON 파싱, 내부 모듈 및 실행 파일 연결, wrapper 인자/경로 테스트와 dry-run만 수행했다. 모델 추론, 학습 수렴, 실제 수치 재현은 검증하지 않았다.

정리 시 Python 13개 파일의 구문 검사, 9개 하위 명령의 CLI 인자 대조, 단위 테스트 8개가 통과했다. 공개 폴더의 고신뢰도 토큰·개인 홈 경로 패턴 검사에는 검출 항목이 없었다. 이는 전체 보안 감사나 실제 GPU 실행 검증을 대신하지 않는다.

## 공개 권한

공개 저장소에 코드를 올릴 때 적용할 라이선스는 저자·공동저자의 결정이 필요하므로 이 정리에서 임의 부여하지 않았다. 데이터와 외부 이미지에 코드 라이선스가 자동 적용되는 것은 아니다. 원본 이미지·질문·API 키·학습 가중치를 포함시키지 않는 기본 구성을 사용한다.
