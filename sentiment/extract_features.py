import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    # 경로 설정
    base_path = 'sentiment'
    input_file = os.path.join(base_path, 'preprocessed_data.csv')
    
    # 저장될 파일 경로
    vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
    features_path = os.path.join(base_path, 'features.pkl')

    # 1. 전처리된 데이터 로드
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run preprocess.py first.")
        return

    print(f"Loading preprocessed data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # 결측치 최종 확인 (토큰화 과정에서 빈 값이 생길 수 있음)
    df.dropna(subset=['tokenized'], inplace=True)
    
    print(f"Total records to vectorize: {len(df)}")

    # 2. TF-IDF + Bigram 설정
    # ngram_range=(1, 2): Unigram(단어 하나)과 Bigram(단어 두 개 조합) 모두 포함
    # max_features: 너무 많은 피처는 메모리 문제를 일으킬 수 있으므로 빈도순 상위 20,000개로 제한 (조정 가능)
    # min_df=2: 최소 2번 이상 등장한 단어만 포함
    print("Extracting features using TF-IDF + Bigram...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_features=20000, 
        min_df=2
    )

    # 3. 피처 추출 (학습)
    # X_tfidf는 희소 행렬(Sparse Matrix) 형태로 저장되어 메모리를 효율적으로 사용합니다.
    X_tfidf = tfidf.fit_transform(df['tokenized'])
    y = df['label'].values

    # 4. 결과 저장
    # Vectorizer: 나중에 새로운 문장을 예측할 때 똑같은 기준으로 변환하기 위해 저장
    # Features: 학습에 바로 사용할 수 있도록 행렬과 라벨 저장
    print("Saving vectorizer and features...")
    joblib.dump(tfidf, vectorizer_path)
    joblib.dump({'X': X_tfidf, 'y': y}, features_path)

    print(f"\nFeature extraction complete!")
    print(f"- Vectorizer saved to: {vectorizer_path}")
    print(f"- Feature matrix saved to: {features_path}")
    print(f"Feature matrix shape: {X_tfidf.shape}")

if __name__ == "__main__":
    main()
