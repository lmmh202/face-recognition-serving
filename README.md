# face-recognition-serving

LFW 얼굴 분류 모델(CNN, PCA+KNN)을 FastAPI로 서빙하는 프로젝트다. 학습 때와 같은 전처리를 서빙에서 그대로 재사용하는 것이 핵심이다.

## 지원 범위와 한계

- LFW 형식 이미지(250×250, JPEG·PNG)만 지원한다.
- 학습된 62명 중 한 명으로만 답하며, 모르는 인물을 거르지 않는다.
- 인물당 20장으로 학습해 test 정확도는 CNN 30.32%, PCA+KNN 22.26%다.

## 실행

[uv](https://docs.astral.sh/uv/)가 필요하다. 모델 파일은 git에 포함되지 않으므로 처음에 학습해야 한다.

```
uv sync
uv run python -m ml.train
uv run uvicorn app.main:app --no-access-log
```

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /predict?model=cnn\|pca-knn` | multipart `file`을 받아 상위 3개 인물과 확률을 반환한다. 기본 모델은 `cnn`이다. |
| `GET /health` | 로드된 모델 목록을 반환한다. |
| `GET /models` | 모델 버전, 라벨 수, 입력 크기, test 정확도를 반환한다. |

```
curl -F file=@face.jpg "http://127.0.0.1:8000/predict?model=cnn"
```

| 상황 | 응답 |
|---|---|
| 디코딩 불가 이미지 | 400 |
| 업로드 5MiB 초과, 2천만 픽셀 초과 | 413 |
| JPEG·PNG가 아닌 형식 | 415 |
| 잘못된 `model` 값, 250×250이 아닌 이미지 | 422 |

업로드 이미지는 저장하거나 로그에 남기지 않는다.

## 테스트

```
uv run pytest tests
```

LFW 데이터나 모델 파일이 없으면 해당 테스트는 건너뛴다. 전처리 검증과 불일치 실험 결과는 [docs/PREPROCESS.md](docs/PREPROCESS.md)에 있다.

## 성능 측정

`uv run python scripts/load_test.py`로 재현한다. 모델과 동시 요청 수 조합마다 LFW 이미지 1장을 300번 요청했다.

| 모델 | 동시 요청 | p50 (ms) | p95 (ms) | 초당 요청 | 에러 |
|---|---|---|---|---|---|
| cnn | 1 | 2.9 | 3.7 | 323.8 | 0 |
| cnn | 8 | 26.3 | 31.1 | 308.9 | 0 |
| pca-knn | 1 | 1.5 | 2.2 | 579.0 | 0 |
| pca-knn | 8 | 8.2 | 11.9 | 903.7 | 0 |

서버 메모리(RSS, 두 모델 로드 상태): 대기 시 547 MB, 최대 573 MB.

측정 환경은 Apple M5(10코어) macOS이고, uvicorn 워커 1개를 띄워 같은 머신의 클라이언트로 요청했다. 네트워크 지연이 없고 CPU도 다르므로 배포 환경의 수치로 보면 안 된다.
