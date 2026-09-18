# VLM Image–Question Mismatch

**VLM 내부 표현을 활용한 이미지–질문 불일치 판별과 선택적 이미지 사용**의 실험 코드를 정리한 저장소입니다.

원본 `vlm_mi_experiments.zip`에서 논문 실험의 실행 경로와 그 의존 모듈을 골랐습니다. 데이터, 모델 가중치, activation 캐시, 개인 실행 로그는 포함하지 않습니다. 이 배포본을 정리하는 과정에서 GPU 실험을 다시 실행하지 않았으며, 아래 명령은 제공된 소스의 실행 절차입니다. 원래 수치와 같은 결과를 확인한 환경 고정본은 아닙니다.

## 포함한 실험

| 논문 내용 | 실행 코드 | 결과 확인 |
|---|---|---|
| Full / Blurred / Shuffled / Text-only 정확도 | `run_mismatch_pipeline.py`, 공통 runner | activation export의 `results.csv`, `cpu_analysis/` |
| 내부 표현·확신도 판별 및 이미지 제외 | `analyze_mismatch_activations.py` | `detector_metrics_frozen.csv`, `routing_metrics_frozen.csv` |
| 관련성 프롬프트 | `b0_direct_query.py`, 보고 프롬프트 V2 | `relevance_v2_summary.json`, JSONL |
| 동일 층 무작위 4-head 20세트 | `analyze_dimension_matched_controls.py` | `layer_matched_random_4head` 행과 summary |
| Qwen3-VL-8B / Qwen2.5-VL-32B 반복 | 동일 mismatch 파이프라인의 모델 인자 변경 | 모델별 출력 폴더 |
| 검색 후보의 별도 판별기 학습 | `retrieval_eval.py`, `analyze_retrieval.py` | `retrieval_probe.json` |
| 검색 없음 / 모두 부착 / 선택적 부착 | `submit_test.py` | 로컬 JSONL / CSV / 답변 JSON |

## 환경

Python 3.10 이상과 해당 모델을 실행할 GPU 환경을 준비합니다. 원본에는 Python 3.11 캐시가 있었지만 정확한 패키지 버전·CUDA·GPU 구성은 기록돼 있지 않았습니다.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt`는 의존성 목록이며 실험 당시 버전 lock이 아닙니다. CUDA에 맞는 PyTorch를 사용하고, 설치한 Transformers가 Qwen3-VL / Qwen2.5-VL 모델 클래스를 제공하는지 확인해야 합니다. 원본의 `transformers>=4.45.0`만으로는 Qwen3-VL 지원을 보장할 수 없어 그 하한을 재현 버전처럼 제시하지 않았습니다.

선택적인 LoRA `--adapter-path` 실행에는 `peft`가 추가로 필요하지만 논문 기본 실행 명령에는 사용하지 않습니다. GPU 선택은 환경에서 설정합니다. 원본 스크립트의 GPU 번호 1 고정은 가져오지 않았습니다.

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py mismatch --profile quick
```

## 데이터 준비

데이터를 이용 가능한 경로에서 별도로 준비합니다.

```text
data/
  train.json
  validation.json
  test.json
  train/                  # 원본 이미지
  validation/
  test/
  train_blurred/          # 미리 생성된 흐림 이미지를 이용할 때
  validation_blurred/
  ablation_manifest_train_portable.csv       # 원래 실험 manifest가 있으면 우선
  ablation_manifest_validation_portable.csv  # 원래 실험 manifest가 있으면 우선
  retrieval_links.json    # 별도 준비한 검색 결과
```

입력은 JSON 배열, JSONL 또는 CSV입니다. 필드와 검색 링크 형식은 [데이터 형식](docs/DATA_FORMAT.md)을 참고하세요. mismatch 파이프라인은 원본 manifest를 우선 사용하며, 없으면 `train.json` / `validation.json`을 읽습니다. 원본 학습 manifest가 없을 때 JSON으로 실행할 수 있게 한 것은 이번 정리에서 추가한 경로 처리입니다. 원래 donor 조합과 행 순서를 재현하려면 당시 manifest가 필요합니다.

논문 핵심 평가는 객관식 학습 518문항·검증 103문항입니다. 검색용 판별기는 별도로 검색 이미지가 확보된 검증 문항에서 학습합니다. 테스트 800문항 전체를 객관식 정확도의 분모로 취급하지 않습니다.

## 실행

저장소 루트에서 실행합니다. `--data-root`와 `--output-root`는 현재 작업 디렉터리 기준 경로를 받으며, 생략하면 저장소의 `data/`, `outputs/`를 사용합니다. `--dry-run`을 붙이면 모델·데이터를 읽지 않고 실행 명령만 확인합니다.

### 1. 학습 및 고정 검증

```bash
python reproduce.py mismatch --profile full
python reproduce.py mismatch --model qwen3-8b --profile full
python reproduce.py mismatch --model qwen25-32b --profile full
```

`quick`은 문항 수와 bootstrap 횟수를 줄이는 동작 확인용이며 논문 수치로 보고하면 안 됩니다. `full`과 출력 경로가 분리됩니다. 논문에 보고하지 않은 permutation ensemble·evidence-first 작업은 배포 설정에서 제거했습니다.

주 모델 출력은 다음과 같습니다.

```text
outputs/qwen3_vl_32b_mismatch_train_full/
  01_mismatch_activation_export/
  cpu_analysis/mismatch_calibration.json
outputs/qwen3_vl_32b_mismatch_validation_full/
  01_mismatch_activation_export/
  cpu_analysis/detector_metrics_frozen.csv
  cpu_analysis/routing_metrics_frozen.csv
```

### 2. 관련성 프롬프트

```bash
python reproduce.py direct-query --split train
python reproduce.py direct-query --split validation
```

보고한 단일 프롬프트(V2)만 실행합니다. 점수는 첫 응답 위치의 `logP(no) - logP(yes)`이며 생성 문장에 대한 응답률이 아닙니다. 원본 파일에는 다른 프롬프트 구현도 남아 있지만 논문 실행 경로에서 사용하지 않습니다.

이 스크립트는 판별 점수를 생성합니다. 관련성 프롬프트의 **학습 임계값을 고정한 이미지 제외 정책 및 내부 표현과의 paired AUROC 차이 집계**를 위한 별도 최종 보고 스크립트는 원본 ZIP에서 확인되지 않았습니다. 그 결과까지 자동 재현된다고 주장하지 않습니다.

### 3. 동일 층 무작위 head 대조

```bash
python reproduce.py controls
```

먼저 동일 모델의 `mismatch`를 실행해야 합니다. 표에 해당하는 대조는 `control_type=layer_matched_random_4head`입니다. 원본은 single-head 추출 후 같은 난수 생성기로 4-head를 뽑으므로, single-head 40회도 유지해 난수 순서를 보존합니다. 4-head 이후 실행되는 projection 반복만 0회로 둡니다. selected-single 및 layer-mean 진단도 저장되므로 모든 출력 행이 논문 보고 항목인 것은 아닙니다. 다른 모델은 `--model`을 함께 지정합니다.

### 4. 검색 후보 다운로드 및 검색용 판별기 학습

```bash
python reproduce.py retrieval-download --links data/retrieval_links.json
python reproduce.py retrieval-fit
```

원본+상위 검색 이미지와 원본+다른 문항의 검색 이미지를 비교합니다. L59–L63의 head 0–7을 쓰는 40-head 구성을 명시했습니다. 최종 판별기는 검색 검증 자료 전체로 재학습하며 `outputs/qwen3_vl_32b_retrieval_validation/cpu_analysis/retrieval_probe.json`에 저장됩니다. 이미지 제외용 4-head 판별기와 서로 다른 학습 결과입니다.

### 5. 테스트 답변 생성

```bash
python reproduce.py retrieval-test --mode baseline
python reproduce.py retrieval-test --mode naive --max-attach 3
python reproduce.py retrieval-test --mode gated --max-attach 1
python reproduce.py retrieval-test --mode gated --max-attach 2
python reproduce.py retrieval-test --mode gated --max-attach 3
```

선택적 부착에는 `youden`을 명시적으로 적용합니다. `max-attach=N`은 상위 N개 후보를 검사한다는 의미이며, N개가 반드시 부착되는 것은 아닙니다. 테스트 정답이 없으면 이 명령만으로 논문의 객관식 정확도를 산출할 수 없습니다.

부착 목록만 추출하려면:

```bash
python extract_attached_images.py \
  --jsonl outputs/qwen3_vl_32b_retrieval_test/gated_n2.jsonl \
  --output outputs/attached_n2.json \
  --csv-output outputs/attached_n2.csv
```

원본 스크립트는 기존 결과를 이어서 실행합니다. 입력, 모델 revision, 프롬프트, seed, 임계값을 바꾸면 **새 `--output-root`**를 사용하세요. 같은 출력 파일을 재사용하면 과거 결과가 섞일 수 있습니다. 검색 링크를 바꿀 때도 새 데이터·이미지 캐시 경로를 사용하세요.

## 공개 범위와 검증

- 파일별 포함·제외 이유와 원본 SHA-256: [source_manifest.json](docs/source_manifest.json)
- 수정 범위와 남아 있는 재현 조건: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- 공통 runner는 다른 탐색 기능도 포함한 원본 모듈을 보존했습니다. 메서드를 임의 삭제해 실행 결과가 달라지는 것을 피하고, 공개 실행 경로·설정만 논문 범위로 제한했습니다.
- `.gitignore`는 데이터·가중치·캐시·실험 결과의 우발적인 추가를 방지합니다. GitHub 게시 및 공개 라이선스 선택은 수행하지 않았습니다.

모델 없이 실행 구조를 확인할 수 있습니다.

```bash
python -m unittest discover -s tests -v
python reproduce.py mismatch --dry-run
```
