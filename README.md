# 말뭉치 모음

개인 연구용으로 수집해둔 말뭉치(코퍼스)들을 공유합니다.

## 구성

- [sentiment](/sentiment)

## 라이센스

Public Domain

## 실행 방법

### 1. 가상환경 생성 (최초 1회)

프로젝트 루트 디렉토리에서 아래 명령어를 입력하여 가상환경을 생성합니다.

```bash
python -m venv .venv
```

### 2. 가상환경 활성화

생성된 가상환경을 활성화합니다.

**Windows:**

```powershell
.\.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. 필수 라이브러리 설치

가상환경이 활성화된 상태에서 필요한 패키지를 설치합니다. (최초 1회)

```bash
pip install -r requirements.txt
```

### 4. 전처리 스크립트 실행

데이터를 정제하고 형태소 분석을 수행하여 `preprocessed_data.csv` 파일을 생성합니다.

```bash
python sentiment/preprocess.py
```

_참고: KoNLPy(Okt) 사용을 위해 Java JDK가 설치되어 있어야 합니다._

### 5. 피처 추출 (TF-IDF + Bigram)
텍스트 데이터를 머신러닝 모델이 이해할 수 있는 수치 행렬로 변환합니다. 계획서의 TF-IDF와 Bigram(단어 2개 조합) 방식이 적용됩니다.

```bash
python sentiment/extract_features.py
```

**생성되는 파일:**
- `tfidf_vectorizer.pkl`: 학습된 단어 사전 및 변환 모델 (새로운 리뷰 예측 시 사용)
- `features.pkl`: 변환된 수치 행렬(X)과 정답 라벨(y) 데이터

### 6. 데이터 검수 (요약)
생성된 `.pkl` 파일의 메타데이터(전체 단어 개수, 데이터 크기, 라벨 분포 등)를 확인합니다.

```bash
python sentiment/check_pkl.py
```

### 7. 샘플 데이터 상세 확인 (DataFrame)
실제 데이터가 어떤 피처(단어)들로 변환되었는지 데이터프레임 형식으로 상세히 확인합니다. 각 리뷰별로 TF-IDF 점수가 높은 상위 5개 단어를 보여줍니다.

```bash
python sentiment/view_samples.py
```

_참고: KoNLPy(Okt) 사용을 위해 Java JDK가 설치되어 있어야 합니다._
