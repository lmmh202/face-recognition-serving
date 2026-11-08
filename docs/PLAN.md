# face-recognition-serving 작업 계획

LFW 얼굴 분류 과제의 CNN과 PCA+KNN 모델을 FastAPI로 서빙하고, Docker와 AWS ECS로 배포한다.

**핵심 인사이트:** 학습 때와 서빙 때 전처리가 달라지면 모델이 틀린다. 학습 때 fit한 PCA 객체와 전처리 설정을 그대로 저장해 재사용한다.

**최소 완성선:** Phase 1~3과 Phase 5.

> 전처리 동작 설명은 아직 라이브러리 소스로 확인하지 않았다. Phase 2의 소스 확인과 golden test로 검증한다.

## 시작 준비

- 과제 코드를 `reference/original_train.py`로 복사한다.
- 학번이 들어간 파일명, PDF 보고서, `.venv`, `figures`는 가져오지 않는다.
- 과제 코드를 공개 레포에 올려도 되는지 수업 규정을 먼저 확인한다.
- Phase 단위로 작업하고 Phase마다 커밋한다.

## 구조

```
app/        main.py  schemas.py  registry.py  logging.py
ml/         preprocess.py  train.py  evaluate.py
artifacts/  cnn-v1/  pca-knn-v1/   # git 제외, 없으면 train.py로 재생성
tests/
Dockerfile
pyproject.toml
README.md
```

## Phase 1: 학습 분리와 모델 저장

- 시각화를 빼고 데이터 로드, 밸런싱, 분할, 학습만 `ml/train.py`로 옮긴다. 시드와 분할 방식은 유지한다.
- PCA+KNN 조합은 test 정확도가 아니라 train 안의 validation이나 CV로 고른다. test는 마지막 평가에만 쓴다.
- KNN은 k≥3으로 제한한다. k=1은 확률이 0 또는 1뿐이라 상위 3개가 의미 없다. k=5면 확률이 0.2 단위로 나온다.
- CNN은 `.keras`로 저장한다.
- PCA와 KNN은 `sklearn.Pipeline` 하나로 묶어 joblib으로 저장한다. 둘이 어긋나는 일을 막기 위해서다.
- 모델마다 `manifest.json`을 둔다.
  - 모델 버전, 라벨 62개, 입력 크기, 전처리 파라미터
  - test 정확도, TF·Keras·sklearn 버전, 시드

**완료 기준:** 저장한 모델을 다시 불러온 test 정확도가 학습 직후와 같다.

## Phase 2: 전처리 공용화 (핵심)

- 학습과 서빙이 같은 `ml/preprocess.py`를 쓴다.
- `fetch_lfw_people`의 기본 동작을 재현한다.
  - funneled 이미지를 `slice_=(70:195, 78:172)`로 자른다.
  - `resize=0.7`로 87×65를 만든다.
- 설치된 sklearn 버전의 소스에서 crop, resize, 흑백 변환의 순서와 보간법을 확인한다.
- 흑백 변환에 주의한다. sklearn은 RGB 채널 평균을 쓰는 것으로 알고 있고, PIL의 `convert("L")`은 가중 평균이라 값이 달라진다.
- Golden test: LFW 원본 JPG를 `preprocess.py`에 넣은 결과가 `fetch_lfw_people` 배열과 허용 오차 안에서 같은지 검사한다.
- 전처리를 일부러 다르게 했을 때 정확도가 얼마나 떨어지는지 측정한다. 자소서에 쓸 실제 수치로 삼는다.

**완료 기준:** golden test를 통과하고, 전처리 불일치 실험 결과를 기록했다.

## Phase 3: FastAPI

### 엔드포인트

- `POST /predict?model=cnn|pca-knn`: 기본값 `cnn`, multipart로 `file`을 받아 상위 3개 인물과 확률을 반환한다.
- `GET /health`: 모델 로드 상태를 반환한다.
- `GET /models`: manifest 요약을 반환한다.

```json
{
  "model": "cnn",
  "model_version": "v1",
  "predictions": [{"name": "...", "probability": 0.62}],
  "inference_ms": 18.4,
  "request_id": "..."
}
```

### 에러 처리

| 상황 | 응답 |
|---|---|
| 디코딩 불가 이미지 | 400 |
| 용량·픽셀 수 초과 | 413 |
| 지원하지 않는 형식 | 415 |
| 잘못된 `model` 값 | 422 |
| 얼굴 미검출 (Phase 4 이후) | 422 |

### 구현 원칙

- 모델은 lifespan에서 시작할 때 한 번만 로드한다.
- 추론은 CPU 작업이므로 엔드포인트를 `async def`가 아닌 `def`로 둔다.
- EXIF 회전을 보정하고, 디코딩 전에 용량과 픽셀 수를 제한한다.
- 요청별 JSON 로그에 `request_id`, 모델과 버전, 전처리·추론 시간, 상태 코드를 남긴다.
- 업로드 이미지는 저장하거나 로그에 남기지 않는다.

## Phase 4: 일반 사진 입력

- v1은 LFW 형식 이미지(250×250)만 정식 지원한다고 문서화한다.
- OpenCV나 MediaPipe로 얼굴을 검출하고, LFW와 비슷한 구도로 확장해 250×250으로 만든 뒤 Phase 2의 파이프라인에 넣는다.
- 모델은 62명 중 하나로만 답하므로, 최고 확률이 임계값보다 낮으면 `unknown`을 반환한다. 임계값은 학습에 없는 LFW 인물로 정한다.

## Phase 5: 테스트와 측정

- pytest로 golden test, 정상 요청, 에러 케이스, 두 모델의 응답 스키마를 검사한다.
- 간단한 부하 테스트로 모델별 p50·p95 지연 시간과 메모리를 측정해 README에 적는다.

## Phase 6: Docker

- `python:3.x-slim`에 `tensorflow-cpu`를 쓰고 버전을 고정한다.
- non-root로 실행한다.
- Mac에서 빌드하면 ARM 이미지가 나온다. `docker buildx --platform linux/amd64`로 빌드하거나 Fargate도 ARM64로 맞춘다.
- 로컬 컨테이너에서 Phase 5의 테스트를 다시 통과시킨다.

## Phase 7: AWS ECS

1. ECR에 이미지를 push한다.
2. VPC를 직접 구성한다. ALB는 public subnet, Fargate 태스크는 private subnet에 두고, 외부 통신은 NAT나 VPC endpoint로 처리한다.
3. 보안 그룹은 ALB→태스크 8000 포트만 허용하고, ALB 접근은 내 IP로 제한한다. 실존 인물의 얼굴을 다루므로 공개 데모로 열어 두지 않는다.
4. Task Definition은 2GB부터 시작해 Phase 5에서 측정한 메모리로 조정한다.
5. 로그는 CloudWatch로 보내고, ALB 헬스체크는 `/health`로 지정한다.
6. 네트워크 구성 과정과 막혔던 지점을 README에 기록한다.
7. 끝나면 ALB와 NAT를 삭제한다. 켜 두는 동안 계속 과금된다.
