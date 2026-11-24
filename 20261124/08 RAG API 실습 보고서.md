# RAG API 실습 보고서

`python main.py`를 실행한 뒤 저장된 `day27_08_live.json`을 살펴봤다. 질문이 API에 전달된 시점부터 답변과 출처가 반환되기까지 각 단계가 어떤 값을 받아 다음 단계에 넘기는지 확인했다.

## 1. 질문 처리 절차

```text
Request Validation
   │  query · 원 질문
   │  history · 대화 이력 목록
   │  trusted_role · 서버가 지정한 접근 역할
   ▼
rewrite
   │  search_query · 검색용 질문
   ▼
multiquery
   │  queries · 검색 질의 목록
   ▼
retrieval
   │  candidates · 검색 후보 문서 목록
   ▼
rerank
   │  ranked · 재순위화된 문서 목록
   ▼
context
   │  context · 모델에 전달할 텍스트
   │  included · 컨텍스트에 포함된 문서 목록
   ▼
generation
   │  status · 답변 상태
   │  result · 답변 결과
   │  sources · 인용 출처 목록
   │  trace · 처리 기록
   ▼
API Response
```

화살표에는 다음 단계로 전달되는 주요 필드만 표시했다.

입력·출력은 Q01 기준이다. 각 단계에서 새로 만들거나 다음 단계에서 사용하는 값만 표시했다.

### 1-1. Request Validation (`RagRequest`)

실습 코드가 준비된 질문을 API에 보낸다. API는 입력을 검증하고 서버의 접근 역할을 붙여 체인을 호출한다.

- **Input** `query="회의실 예약을 취소하는 방법은?"`, `category="시설"`, `candidate_k=5`, `context_k=2`
- **Output** 검증된 요청과 `trusted_role="learner"`

### 1-2. Query Rewrite (`rewrite`)

- **Input** `query`, `history`, `use_rewrite=True`
- **Output** `search_query="회의실 예약을 취소하는 방법은?"` (원 질문과 동일)

### 1-3. Multi-Query (`multiquery`)

- **Input** `search_query`, `use_multiquery=False`
- **Output** `queries=["회의실 예약을 취소하는 방법은?"]`

### 1-4. Retrieval (`retrieval`)

카테고리와 접근 권한을 적용해 후보 문서를 검색한다.

- **Input** `queries`, `category`, `trusted_role`, `candidate_k=5`
- **Output** `candidates` 5개. `candidate_ids`의 상위 3개는 `FAC-102-C1` → `FAC-107-C1` → `FAC-109-C1`

### 1-5. Rerank (`rerank`)

검색 질의와 후보 문서의 관련도를 다시 평가한다.

- **Input** `search_query`, `candidates` 5개, `use_rerank=True`
- **Output** `ranked` 5개. `ranked_ids`의 상위 3개는 `FAC-102-C1` → `FAC-109-C1` → `FAC-107-C1`

### 1-6. Context (`context`)

재순위화된 문서에서 최대 `context_k`개를 골라 모델에 전달할 컨텍스트를 구성한다.

- **Input** `ranked`, `context_k=2`
- **Output** `context`, `included`, `included_ids`. Q01의 `included_ids`는 `["FAC-102-C1", "FAC-109-C1"]`

### 1-7. Generation (`generation`)

컨텍스트를 바탕으로 답변을 생성하고, 인용 ID를 검증해 응답 본문을 구성한다.

- **Input** `query`, `search_query`, `context`, `included`
- **Output** `status`, `result`, `sources`, `trace`를 담은 응답 딕셔너리

### 1-8. API Response

FastAPI가 체인의 반환값을 HTTP 응답 본문으로 보낸다. 실습 코드는 HTTP 상태와 본문을 JSON에 저장한다.

- **Input** `generation`이 반환한 응답 딕셔너리
- **Output** HTTP 200과 JSON 응답 본문. 저장 파일에서는 각각 `http_status`와 `body`에 기록

## 2. 실행 결과와 응답 해석

### 2-1. Response Structure

저장된 실행 결과 기준이다.

```text
rows[]                           실행 사례 목록
├─ question_id                   사례 ID
├─ http_status                   HTTP 상태 코드
└─ body                          HTTP 응답 본문
   ├─ HTTP 200인 경우
   │   ├── status                    result.status에서 가져온 답변 상태
   │   ├── result                    답변 객체
   │   │   ├── status                    답변 상태
   │   │   ├── answer                    답변 문장
   │   │   ├── source_ids[]              최종 답변의 인용 번호
   │   │   └── missing_information       답변에 필요한 추가 정보
   │   ├── sources[]                 인용 번호에 대응하는 출처 정보
   │   └── trace                     검색 질의, 단계별 문서 ID, 소요 시간
   └─ HTTP 422인 경우
       └── detail[]                  입력 검증 오류 목록
```

### 2-2. Q01 (`answered`)

- **HTTP** `200`
- **Answer** 예약 시스템의 ‘나의 예약’ 메뉴에서 취소를 선택하면 된다. 시작 2시간 전까지 취소할 수 있다.
- **Source** `result.source_ids=[2]` → `sources[0].citation_id=2` → `FAC-102-C1` (`2026-11-v2`)

인용 문서 `FAC-102-C1`의 본문에서 취소 메뉴와 시간 조건을 모두 확인했다.

### 2-3. Q06 (`insufficient_context`)

- **HTTP** `200`
- **Result** `status="insufficient_context"`, `source_ids=[]`
- **Missing information** “과제별 안내에 있는 정확한 마감 날짜가 필요합니다.”

문서 2개가 컨텍스트에 포함됐지만, 정확한 마감일을 답할 근거가 부족해 `insufficient_context`가 반환됐다.

### 2-4. INVALID_K (`422`)

- **HTTP** `422`
- **Error field** `detail[0].loc=["body", "candidate_k"]`
- **Rejected value** `detail[0].input=0` · 최소 허용값 `detail[0].ctx.ge=1`

요청이 입력 검증에서 거절되어 체인이 호출되기 전에 처리가 종료됐다.

## 3. 결론

검색 질의로 후보 문서를 찾고, 재순위화와 컨텍스트 구성을 거쳐 답변과 출처를 반환하는 흐름을 확인했다. 저장된 세 사례를 통해 근거를 인용한 답변, 근거가 부족하다는 응답, 잘못된 입력에 대한 오류가 각각 어느 단계에서 결정되는지 확인할 수 있었다.
