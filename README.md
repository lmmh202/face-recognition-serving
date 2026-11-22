# face-recognition-serving

LFW 얼굴 분류 모델(CNN, PCA+KNN)을 FastAPI로 서빙하는 프로젝트다. 학습 때와 같은 전처리를 서빙에서 그대로 재사용하는 것이 핵심이다.

> LFW 데이터셋의 공인 62명 중 업로드한 얼굴과 가장 닮은 사람을 고르는 데모다. 신원 확인 용도로 쓰면 안 되며, 학습에 없는 사람의 사진에도 62명 중 한 명의 이름을 돌려줄 수 있다.

데모: <https://face-recognition-serving.hyeong.space/docs> (요청이 없으면 서버가 꺼져 있어 첫 요청에 1분 정도 걸린다)

## 지원 범위와 한계

- JPEG·PNG 사진에서 가장 큰 얼굴 하나를 찾아 학습된 62명 중 한 명으로 분류한다.
- LFW 형식 입력의 test 정확도는 인물당 20장으로 학습한 CNN이 30.32%, PCA+KNN이 22.26%다.
- `cnn-v2`는 전체 사진과 증강, 검출 crop으로 학습했으며 얼굴 검출 경로의 test 정확도가 58.39%다(CNN 30.65%, PCA+KNN 14.52%).
- 최고 확률이 임계값보다 낮으면 `identity`가 `unknown`이다. 임계값이 높아 학습된 인물도 대부분 `unknown`이 된다.
- Wikimedia Commons의 실제 사진 287장(62명)에서 top-1은 cnn-v2 23.34%, CNN 10.45%, PCA+KNN 11.50%다. LFW에서보다 크게 낮다. 자세한 내용은 [docs/REAL_PHOTOS.md](docs/REAL_PHOTOS.md)에 있다.

## 실행

[uv](https://docs.astral.sh/uv/)가 필요하다. 모델 파일은 git에 포함되지 않으므로 처음에 학습해야 한다.

```
uv sync
uv run python -m ml.train
uv run python -m ml.train_cnn_v2
uv run python -m ml.calibrate
uv run uvicorn app.main:app --no-access-log
```

## Docker

이미지는 `artifacts/`의 모델을 포함하므로, 빌드 전에 학습이 끝나 있거나 모델 Release를 받아 둬야 한다. non-root로 실행된다. 운영(OCI)은 `linux/arm64`, 예전 Azure 배포는 `linux/amd64`다. x86 Linux에서는 `tensorflow-cpu`, 그 밖에서는 `tensorflow`가 설치되어 arm64 이미지가 더 크다(약 2.3GB).

```
docker build --target runtime -t face-recognition-serving:local .
docker run --rm -p 8000:8000 face-recognition-serving:local
```

컨테이너 안에서 테스트하려면 `--target test`로 빌드한다. LFW 데이터를 마운트해야 golden test와 정확도 검증까지 실행된다.

```
docker build --target test -t face-recognition-serving:test .
docker run --rm -v ~/scikit_learn_data:/data:ro -e SCIKIT_LEARN_DATA=/data face-recognition-serving:test
```

## 배포 (OCI)

운영 환경은 공용 인프라 레포 `lmmh202/infra`의 OCI A1 인스턴스(`minhyeong-cloud`, arm64, 2 OCPU / 12GB)다. Dockge 스택 `/opt/stacks/face-recognition-serving`에서 컨테이너 1개로 돌고, Caddy가 HTTPS를 맡는다.

| 파일 | 역할 |
|---|---|
| `.github/workflows/deploy.yml` | `test`가 `main`에서 통과하면 인스턴스의 self-hosted runner(`face-recognition-serving` 라벨)가 모델 Release를 받아 이미지를 빌드하고 배포한다. 레지스트리는 쓰지 않는다 |
| `deploy/compose.yaml` | 스택 정의. `proxy` 네트워크, `/health` 검사, 메모리 1GB / CPU 1개 제한. 컨테이너 1개가 비용과 남용 상한이다 |
| `deploy/deploy.sh` | `.env`에 태그를 쓰고 `docker compose up --wait` 후 `.deployed-tag`에 기록한다 |
| `.github/models.env` | 모델 Release 이름과 SHA-256. `test`와 `deploy`가 함께 읽는다 |

`infra` 쪽에는 `server/proxy/sites/face-recognition-serving.caddy`와 `terraform/cloudflare`의 A 레코드가 있다.

### 처음 설정

```
# infra 레포에서
scripts/install-runner.sh lmmh202/face-recognition-serving face-recognition-serving
ssh oci-osaka 'sudo mkdir -p /opt/stacks/face-recognition-serving && sudo chown github-runner:github-runner /opt/stacks/face-recognition-serving'
scripts/deploy-proxy.sh
```

레포가 공개라서, 레포 설정의 fork PR 워크플로 승인 정책을 "모든 외부 기여자 승인 필요"(`all_external_contributors`)로 둔다. runner는 docker 그룹이라 서버의 root 권한과 같으므로, 외부 PR의 Actions를 승인하기 전에 워크플로 변경(특히 `runs-on`)을 확인한다.

그다음 Actions에서 `deploy`를 수동 실행(`workflow_dispatch`)하고, 동작을 확인한 뒤 `terraform/cloudflare`를 apply해 DNS를 옮긴다.

### 롤백

`deploy`를 수동 실행하면서 `tag`에 서버에 남아 있는 이전 태그(최근 3개 보관)를 넣는다. 현재 태그는 서버의 `.deployed-tag`에 있다.

## 이전 배포 (Azure Container Apps)

OCI로 옮기기 전 배포다. 리소스는 당분간 남겨 두며, DNS를 옮긴 뒤에는 `azurecontainerapps.io` 주소로만 접근된다.

인프라는 `infra/`의 Terraform 코드로 관리한다. 리전은 Azure for Students 구독 정책에 따라 `koreacentral`이다.

| 리소스 | 설정 |
|---|---|
| Container Registry | Basic, 관리자 계정 비활성 |
| Container Apps 환경 | `environmentMode = WorkloadProfiles`, `Consumption` 프로필만 사용 |
| Container App | 1 vCPU / 2Gi, 레플리카 0~1, HTTPS만 허용, `/health` 검사 |
| 관리 ID + `AcrPull` | 비밀번호 없이 ACR에서 이미지를 받는다 |
| Log Analytics | 30일 보관, 하루 0.5GB 상한 |

### 처음 배포

```
cd infra
terraform init
export ARM_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
terraform apply -target=azurerm_container_registry.app   # 이미지를 올릴 ACR 먼저

cd ..
TAG=$(git rev-parse --short HEAD)
ACR=crfacerecognitionserving.azurecr.io
docker buildx build --platform linux/amd64 --load -t $ACR/face-recognition-serving:$TAG .
az acr login -n crfacerecognitionserving
docker push $ACR/face-recognition-serving:$TAG

cd infra
echo "image_tag = \"$TAG\"" > terraform.tfvars   # git에서 제외된 파일
terraform apply
```

새 이미지를 배포할 때는 빌드와 push 후 `terraform.tfvars`의 태그만 바꾸고 `terraform apply`한다.

### 사용자 지정 도메인

도메인과 관리형 인증서는 Terraform 밖에서 CLI로 연결했다. 환경을 새로 만들면 다시 해야 한다.

1. DNS(Spaceship)에 레코드 2개를 등록한다. 값은 `terraform output`으로 확인한다.

    | 종류 | Host | 값 |
    |---|---|---|
    | CNAME | `face-recognition-serving` | `app_url`의 호스트 이름 |
    | TXT | `asuid.face-recognition-serving` | `domain_verification_id` |

2. 도메인을 등록하고 무료 인증서를 연결한다. 발급에 몇 분 걸린다.

    ```
    az containerapp hostname add --hostname face-recognition-serving.hyeong.space \
      -g rg-face-recognition-serving -n ca-face-recognition-serving
    az containerapp hostname bind --hostname face-recognition-serving.hyeong.space \
      -g rg-face-recognition-serving -n ca-face-recognition-serving \
      --environment cae-face-recognition-serving --validation-method CNAME
    ```

### 비용

요청이 없으면 레플리카가 0대라 앱 비용이 없고, 깨어난 동안만 소비형 요금이 붙는다. 매달 무료량(18만 vCPU초, 36만 GiB초, 요청 200만 건)이 있어 데모 수준에서는 대부분 ACR 요금(하루 약 $0.17, 한 달 약 $5)만 나간다. 최대 레플리카 1개가 남용 시 비용 상한이다.

### 막혔던 지점

| 문제 | 원인 | 해결 |
|---|---|---|
| Container Apps에서 이미지 실행 불가 | Container Apps는 `linux/amd64`만 지원 | ARM 이미지를 amd64로 전환하고 x86용 `tensorflow-cpu` 사용 |
| `az acr build` 실패 (`TasksOperationsNotAllowed`) | 학생 구독에서 ACR Tasks 차단 | 맥에서 에뮬레이션으로 빌드해 push |
| 관리 ID 생성 실패 (`MissingSubscriptionRegistration`) | `Microsoft.ManagedIdentity` 공급자 미등록 | 공급자 등록 목록에 추가 |
| 환경 프로필 변경 `plan` | Azure가 `Consumption` 프로필을 자동 추가 | 코드에 프로필 명시 |
| 도메인 등록 실패 (`ExpressCustomDomainsNotSupported`) | 모드를 지정하지 않자 Azure가 Express 환경을 만듦. Express는 사용자 지정 도메인 미지원. `azurerm`으로는 모드를 지정할 수 없음 | `azapi`로 `environmentMode = WorkloadProfiles`를 지정해 환경 재생성 |

### 0대에서 깨어나는 시간

| 환경 | 첫 `/health` 응답 | 이후 `/predict` |
|---|---|---|
| Express | 13.9초 | 0.24초 |
| WorkloadProfiles (현재) | 64.6초 | 0.22~0.37초 |

각 1회 측정이다. 도메인을 쓰려고 Express를 포기한 대가로 첫 요청이 느려졌다. 이미지(약 532MB)를 받고 TensorFlow와 모델 세 개를 올리는 시간이 대부분이다.

### 삭제

```
cd infra && terraform destroy
```

DNS는 이미 OCI를 가리키므로 따로 지울 레코드는 없다.

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /predict?model=cnn\|cnn-v2\|pca-knn&detect=true\|false` | multipart `file`을 받아 판정 결과(`identity`)와 상위 3개 인물·확률을 반환한다. 기본 모델은 `cnn-v2`다. `detect=false`는 얼굴 검출을 건너뛰며 LFW 형식(250×250)만 받는다. |
| `GET /health` | 로드된 모델 목록을 반환한다. |
| `GET /models` | 모델 버전, 라벨 수, 입력 크기, test 정확도, unknown 임계값을 반환한다. |

```
curl -F file=@face.jpg "https://face-recognition-serving.hyeong.space/predict"
curl -F file=@face.jpg "http://127.0.0.1:8000/predict?model=cnn"   # 로컬
```

| 상황 | 응답 |
|---|---|
| 디코딩 불가 이미지 | 400 |
| 업로드 5MiB 초과, 2천만 픽셀 초과 | 413 |
| JPEG·PNG가 아닌 형식 | 415 |
| 잘못된 `model` 값, 얼굴 미검출, `detect=false`인데 250×250이 아닌 이미지 | 422 |

업로드 이미지는 저장하거나 로그에 남기지 않는다.

## 테스트

```
uv run pytest tests
```

LFW 데이터나 모델 파일이 없으면 해당 테스트는 건너뛴다. 전처리 검증과 불일치 실험 결과는 [docs/PREPROCESS.md](docs/PREPROCESS.md)에 있다.

### CI

`.github/workflows/test.yml`이 `main` push와 PR마다 전체 테스트를 실행한다. LFW는 처음 한 번 내려받아 캐시하고, 모델은 GitHub Release `models-2026-11-08`의 `artifacts.tar.gz`를 SHA-256으로 확인한 뒤 쓴다. CI는 `REQUIRE_TEST_DATA=1`로 실행하므로 LFW나 모델이 없으면 테스트를 건너뛰지 않고 실패한다.

모델을 다시 학습했다면 새 Release를 올리고 workflow의 `MODELS_RELEASE`와 `MODELS_SHA256`을 바꾼다.

```
COPYFILE_DISABLE=1 tar --no-xattrs -czf artifacts.tar.gz artifacts
shasum -a 256 artifacts.tar.gz
gh release create models-YYYY-MM-DD artifacts.tar.gz --title "Models YYYY-MM-DD"
```

## 성능 측정

`uv run python scripts/load_test.py`로 재현한다. 조합마다 LFW 이미지 1장을 300번 요청했다.

| 모델 | 얼굴 검출 | 동시 요청 | p50 (ms) | p95 (ms) | 초당 요청 | 에러 |
|---|---|---|---|---|---|---|
| cnn | off | 1 | 3.1 | 5.4 | 295.4 | 0 |
| cnn | off | 8 | 26.0 | 32.3 | 307.8 | 0 |
| cnn | on | 1 | 5.3 | 6.3 | 181.8 | 0 |
| cnn | on | 8 | 29.1 | 37.7 | 273.3 | 0 |
| cnn-v2 | off | 1 | 3.3 | 5.1 | 284.2 | 0 |
| cnn-v2 | off | 8 | 30.7 | 36.4 | 261.0 | 0 |
| cnn-v2 | on | 1 | 5.6 | 6.3 | 176.6 | 0 |
| cnn-v2 | on | 8 | 34.4 | 47.4 | 226.0 | 0 |
| pca-knn | off | 1 | 1.6 | 2.1 | 572.4 | 0 |
| pca-knn | off | 8 | 8.5 | 12.7 | 883.9 | 0 |
| pca-knn | on | 1 | 4.1 | 7.5 | 212.9 | 0 |
| pca-knn | on | 8 | 18.3 | 21.8 | 427.8 | 0 |

서버 메모리(RSS, 세 모델 로드 상태): 대기 시 621 MB, 최대 648 MB.

측정 환경은 Apple M5(10코어) macOS이고, uvicorn 워커 1개를 띄워 같은 머신의 클라이언트로 요청했다. 네트워크 지연이 없고 CPU도 다르므로 배포 환경의 수치로 보면 안 된다.
