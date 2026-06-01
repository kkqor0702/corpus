## 구성

- [sentiment](/sentiment) : 리뷰 감성 분류 + 부정 리뷰 군집화. 
- 폴더 내 파일 구성 상세는 [sentiment/README.md](sentiment/README.md) 참고.


<br>

## 실행 방법

> 모든 명령은 **프로젝트 루트 디렉토리(이 README가 있는 폴더)에서** 실행합니다. 스크립트 내부 경로가 `sentiment/...` 기준으로 잡혀 있습니다.

### 1. 가상환경 생성 (최초 1회)

```bash
python -m venv .venv
```

### 2. 가상환경 활성화

**Windows:**

```powershell
.\.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. 필수 라이브러리 설치 (최초 1회)

```bash
pip install -r requirements.txt
```

_참고: KoNLPy(Okt) 사용을 위해 Java JDK가 설치되어 있어야 합니다._

### 4. 전처리

데이터를 정제·토큰화·라벨링하여 `preprocessed_data.csv`를 생성합니다.

```bash
python sentiment/preprocess.py
```

### 5. 피처 추출 (TF-IDF + Bigram)

텍스트를 수치 행렬로 변환합니다. → `tfidf_vectorizer.pkl`, `features.pkl` 생성

```bash
python sentiment/extract_features.py
```

### 6. (선택) 데이터 검수

생성된 `.pkl`의 메타데이터(전체 단어 개수, 데이터 크기, 라벨 분포 등) 확인:

```bash
python sentiment/check_pkl.py
```

각 리뷰가 어떤 피처로 변환됐는지 데이터프레임으로 상세 확인 (TF-IDF 상위 5개 단어):

```bash
python sentiment/view_samples.py
```

### 7. 감성 분류 모델 학습 및 평가

`features.pkl`을 7:3으로 분할해 Logistic Regression 모델을 학습·평가합니다. (Confusion Matrix, Precision/Recall/F1 출력) → `model.pkl` 생성

```bash
python sentiment/train_model.py
```

### 8. 부정 리뷰 군집화

`model.pkl`로 부정 리뷰를 골라내 TruncatedSVD(100차원) + K-Means(5개)로 군집화하고, 군집별 대표 키워드·워드클라우드·빈도로 불만 유형을 분석합니다.

```bash
jupyter notebook sentiment/negative_review_clustering.ipynb
```

---

## 의존 관계 요약

```
naver_shopping.txt
        │  preprocess.py
        ▼
preprocessed_data.csv
        │  extract_features.py
        ▼
tfidf_vectorizer.pkl + features.pkl
        │  train_model.py
        ▼
     model.pkl
        │  negative_review_clustering.ipynb
        ▼
  부정 리뷰 5개 불만 유형 군집 + 키워드/워드클라우드/빈도 분석
```

각 단계는 앞 단계의 산출물(`.csv` / `.pkl`)을 입력으로 받으므로 **순서대로** 실행해야 합니다.
