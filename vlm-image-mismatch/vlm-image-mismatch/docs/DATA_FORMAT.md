# 입력 형식

아래는 형식 설명용 가상 예시이며 실제 말뭉치나 실험 표본이 아닙니다.

## QA 입력

JSON 최상위는 배열입니다. CSV에서도 같은 열 이름을 쓸 수 있습니다.

```json
[
  {
    "question_id": "example-001",
    "question_form": "MC",
    "question": "그림에 표시된 도형은 무엇인가?",
    "options": ["원", "삼각형", "사각형", "별"],
    "answer": "1",
    "image_name": "example-001.jpg",
    "condition": "full",
    "split": "train"
  }
]
```

`question_form`은 MC/SA/LA를 구분합니다. 핵심 mismatch 평가는 MC를 사용합니다. 선택지 목록은 CSV에서 `options_json` 열에 JSON 배열로 넣을 수 있습니다. 코드의 `canonical_row`는 `metadata`, `model_input`, `model_output`으로 중첩된 원본 형식도 지원합니다.

이미지 파일은 `--image-root` 아래에 두며 이름으로 검색됩니다. 이미지 이름과 문항 ID는 분할 내에서 모호하지 않아야 합니다. Shuffled 생성에는 다른 문항의 다른 이미지가 필요하므로 최소 두 개의 서로 다른 문항·이미지가 있어야 합니다.

이미 만든 교란 manifest가 있다면 `condition`, `input_image_variant`, `input_image_name` 열을 유지하세요. 교란 이미지를 생성하는 방식과 입력 행 집합이 달라지면 동일 seed라도 원래의 donor 쌍을 보장하지 못합니다. `strict_length_match` 검사를 해제하지 말고 export의 실제 토큰 수를 확인하세요.

## 검색 결과 링크

`fetch_retrieved_images.py`는 분할 → 문항 ID → 순위순 후보 배열을 받습니다.

```json
{
  "validation": {
    "example-001": [
      {
        "출처": "example",
        "표제어": "예시 항목",
        "설명": "형식 설명용 문자열",
        "유사도": 0.8,
        "이미지링크": "https://example.org/image.jpg",
        "페이지링크": "https://example.org/item"
      }
    ]
  },
  "test": {}
}
```

배열 순서가 후보 순위입니다. 예시 URL은 실제 다운로드용 자료가 아닙니다. 다운로드 결과는 `retrieved_manifest.csv`와 `retrieved_images/`로 저장됩니다. 질문 ID·분할이 QA 입력과 일치해야 합니다. 검색용 코드가 생성하는 manifest 열은 split, question_id, rank, source, title, description, similarity, image_url, page_url 및 다운로드 상태/이미지 경로입니다.
