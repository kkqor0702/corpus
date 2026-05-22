import joblib
import os
import pandas as pd
import numpy as np

def view_samples(num_samples=5):
    """
    지정된 개수만큼의 데이터를 데이터프레임 형태로 출력하여
    원래 라벨과 추출된 주요 피처(단어)들을 한눈에 확인합니다.
    """
    base_path = 'sentiment'
    vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
    features_path = os.path.join(base_path, 'features.pkl')

    if not os.path.exists(vectorizer_path) or not os.path.exists(features_path):
        print("필요한 .pkl 파일이 없습니다. 먼저 extract_features.py를 실행해주세요.")
        return

    print(f"=== 상위 {num_samples}개 데이터 샘플 상세 분석 ===")
    
    # 데이터 로드
    tfidf = joblib.load(vectorizer_path)
    data = joblib.load(features_path)
    X = data['X']
    y = data['y']
    
    feature_names = tfidf.get_feature_names_out()
    
    results = []
    
    # 지정된 개수만큼 반복
    for i in range(min(num_samples, X.shape[0])):
        row = X[i]
        label = "긍정(1)" if y[i] == 1 else "부정(0)"
        
        indices = row.indices
        values = row.data
        
        if len(indices) == 0:
            top_features = "없음"
        else:
            # (단어, 점수) 쌍으로 정렬하여 상위 5개만 문자열로 묶기
            feature_scores = [(feature_names[idx], val) for idx, val in zip(indices, values)]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            top_features = ", ".join([f"{word}({score:.2f})" for word, score in feature_scores[:5]])
            
        results.append({
            "인덱스": i,
            "라벨": label,
            "주요 피처 및 TF-IDF 점수 (Top 5)": top_features
        })
        
    # 데이터프레임 생성 및 출력
    df_results = pd.DataFrame(results)
    
    # 컬럼 정렬 및 보기 좋게 출력
    pd.set_option('display.max_colwidth', None) # 텍스트 잘리지 않게 설정
    pd.set_option('display.unicode.east_asian_width', True) # 한글 줄맞춤
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    # 원하는 샘플 개수를 여기에 입력하세요.
    view_samples(num_samples=100)
