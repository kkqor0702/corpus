# sentiment — 파일 구성

네이버 쇼핑 리뷰로 **감성 분류 모델**을 만들고, 그 모델이 **부정으로 예측한 리뷰만 골라 군집화**해 불만 유형을 분석하는 프로젝트입니다.

이 문서는 폴더 안 각 파일이 무엇인지 정리한 것입니다. **실행 방법·순서는 루트 [README.md](../README.md) 참고.**

<br>

## 데이터셋

`naver_shopping.txt`는 간단한 감성 분류 실험을 위해 수집된 공개 말뭉치입니다. (출처: [bab2min/corpus](https://github.com/bab2min/corpus/tree/master/sentiment))

| 항목 | 내용 |
|---|---|
| 출처 | 네이버 쇼핑 (2020.06~07) |
| 건수 | 20만 건 (긍정 99,963 / 부정 100,037, 약 1:1) |
| 형식 | `별점(1~5)\t텍스트` (탭 구분) |
| 라벨 기준 | 긍정 = 4~5점 / 부정 = 1~2점 (3점은 제외) |

<br>

## 파일 구성

### 실행 스크립트 (파이프라인 단계)

#### `preprocess.py` — 전처리
원본 리뷰를 정제·토큰화·라벨링합니다.
- **정제**: 정규식 `[^가-힣a-zA-Z\s]`로 특수문자·이모지·숫자 등 노이즈 제거
- **토큰화**: KoNLPy `Okt`로 명사/동사/형용사 추출(`stem=True`), 불용어(은/는/이/가 등) 제거
- **정제**: 중복 리뷰 제거(`drop_duplicates`), 토큰화 후 빈 행 제거
- **라벨링**: 별점 기반 이진화 (긍정 4~5 / 부정 1~2, 3점 제외)
- **메모리**: 청크(5,000건) 단위로 처리
- 입력: `naver_shopping.txt` → 출력: `preprocessed_data.csv`

#### `extract_features.py` — 피처 추출
토큰화된 텍스트를 수치 행렬로 변환합니다.
- **TF-IDF + Bigram**: `ngram_range=(1,2)`, `max_features=20000`, `min_df=2`
- Bigram으로 "안 좋다", "배송 느리다" 같은 두 단어 조합의 맥락을 보존
- 입력: `preprocessed_data.csv` → 출력: `tfidf_vectorizer.pkl`, `features.pkl`

#### `train_model.py` — 감성 분류 모델 학습
- 피처 행렬을 7:3으로 분할해 **로지스틱 회귀** 모델 학습
- 혼동행렬(Confusion Matrix)·분류 리포트(Precision/Recall/F1) 출력
- 입력: `features.pkl` → 출력: `model.pkl`

#### `negative_review_clustering.ipynb` — 부정 리뷰 군집화
- `model.pkl`로 전체 리뷰의 긍/부정을 예측
- **부정으로 예측된 리뷰만** TruncatedSVD(100차원)로 축소 후 K-Means(5개 군집)
- 군집별 대표 키워드·워드클라우드·빈도 분석으로 불만 유형 파악
- 입력: `model.pkl`, `tfidf_vectorizer.pkl`, `features.pkl`

### 점검용 보조 스크립트 (파이프라인 필수 아님)

#### `check_pkl.py`
`tfidf_vectorizer.pkl`·`features.pkl` 내용 확인 — 피처(단어) 개수, 라벨 분포, 첫 샘플의 TF-IDF 점수 상위 단어 출력.

#### `view_samples.py`
데이터 샘플별 라벨과 주요 피처(TF-IDF 상위 5개 단어)를 데이터프레임 표로 확인.

### 데이터 / 산출물 파일

| 파일 | 종류 | 설명 |
|---|---|---|
| `naver_shopping.txt` | 원본 | 네이버 쇼핑 리뷰 20만 건 (`별점\t텍스트`) |
| `preprocessed_data.csv` | 중간 산출물 | 전처리 결과 (`label,tokenized`) |
| `tfidf_vectorizer.pkl` | 산출물 | 학습된 TF-IDF 벡터라이저 (새 문장 변환용) |
| `features.pkl` | 산출물 | 피처 행렬+라벨 `{'X', 'y'}` (희소행렬) |
| `model.pkl` | 산출물 | 학습된 로지스틱 회귀 감성 분류 모델 |
