import joblib
import os
import numpy as np

def check_pkl():
    base_path = 'sentiment'
    vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
    features_path = os.path.join(base_path, 'features.pkl')

    print("=== 1. TF-IDF Vectorizer 확인 ===")
    tfidf = None
    if os.path.exists(vectorizer_path):
        tfidf = joblib.load(vectorizer_path)
        vocab = tfidf.vocabulary_
        print(f"추출된 전체 피처(단어) 개수: {len(vocab)}")
        
        # 샘플 단어 (가나다순)
        sorted_vocab = sorted(vocab.items())
        print("샘플 단어 (10개):", [word for word, idx in sorted_vocab[:10]])
        print(f"N-gram 범위: {tfidf.ngram_range}")
    else:
        print(f"File not found: {vectorizer_path}")

    print("\n" + "="*30 + "\n")

    print("=== 2. Feature Matrix & Labels 확인 ===")
    if os.path.exists(features_path):
        data = joblib.load(features_path)
        X = data['X']
        y = data['y']
        
        print(f"행렬 크기 (데이터 개수, 피처 개수): {X.shape}")
        
        # 라벨 분포 (np.int64 제거하고 깔끔하게 출력)
        unique, counts = np.unique(y, return_counts=True)
        print("라벨 분포:")
        for label, count in zip(unique, counts):
            label_name = "긍정" if int(label) == 1 else "부정"
            print(f"  - {label_name}({int(label)}): {int(count)}개")
        
        # 상세 분석 (텍스트와 피처 매칭)
        if tfidf is not None:
            print(f"\n[샘플 데이터 상세 분석 (첫 번째 데이터)]")
            feature_names = tfidf.get_feature_names_out()
            
            # 첫 번째 행의 데이터 추출
            first_row = X[0]
            indices = first_row.indices
            values = first_row.data
            
            if len(indices) == 0:
                print("추출된 피처가 없습니다 (토큰화 결과가 비어있을 수 있음).")
            else:
                # (단어, 점수) 쌍 생성 및 정렬
                feature_scores = [(feature_names[idx], val) for idx, val in zip(indices, values)]
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                
                print(f"추출된 피처 및 TF-IDF 점수 (상위 10개):")
                for word, score in feature_scores[:10]:
                    print(f"  - {word}: {score:.4f}")
        else:
            print("\nVectorizer를 불러오지 못해 상세 분석을 수행할 수 없습니다.")
    else:
        print(f"File not found: {features_path}")

if __name__ == "__main__":
    check_pkl()
