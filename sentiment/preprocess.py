import pandas as pd
import re
from konlpy.tag import Okt
import os
import gc

# --- 설정 ---
# 테스트하고 싶은 데이터 개수 입력 (전체 데이터를 하려면 None 설정)
SAMPLE_SIZE = 12000 
CHUNK_SIZE = 5000  # 메모리 절약을 위해 한 번에 처리할 데이터 양
# -----------

# 1. 환경 설정 및 초기화
okt = Okt()

# 불용어 리스트
STOPWORDS = set(['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한', '을', '를', '에', '와', '과', '도'])

def clean_text(text):
    """
    텍스트 정제: 한글, 영문, 공백 제외 제거
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """
    형태소 분석: 명사 추출 및 동사/형용사 어간 추출
    """
    if not text:
        return ""
    morphs = okt.pos(text, stem=True)
    words = [word for word, pos in morphs if pos in ['Noun', 'Verb', 'Adjective']]
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

def process_chunk(chunk, seen_reviews, data_type='naver'):
    """
    데이터프레임 청크 단위 전처리 (전체 중복 체크 포함)
    """
    if data_type == 'naver':
        chunk.columns = ['rating', 'review']
        chunk = chunk[chunk['rating'] != 3].copy()
        chunk['label'] = chunk['rating'].apply(lambda x: 1 if x >= 4 else 0)
    else:
        chunk.columns = ['label', 'review']
    
    chunk = chunk[['review', 'label']]
    
    # 1. 텍스트 정제
    chunk['review'] = chunk['review'].apply(clean_text)
    
    # 2. 결측치 제거
    chunk = chunk[chunk['review'].str.strip() != '']
    chunk.dropna(subset=['review'], inplace=True)

    # 3. 중복 제거 (현재 청크 내 중복 + 이전 배치들과의 중복)
    # 현재 청크 내부 중복 제거
    chunk.drop_duplicates(subset=['review'], inplace=True)
    # 이전 배치의 데이터와 비교하여 중복 제거
    chunk = chunk[~chunk['review'].isin(seen_reviews)]
    
    # 4. 토큰화
    chunk['tokenized'] = chunk['review'].apply(tokenize)
    
    # 5. 토큰화 후 빈 결과 제거
    chunk = chunk[chunk['tokenized'].str.strip() != '']
    
    # 새로운 데이터 seen_reviews에 업데이트
    seen_reviews.update(chunk['review'].tolist())
    
    return chunk[['label', 'tokenized']]

def main():
    # 경로 설정
    base_path = 'sentiment'
    naver_file = os.path.join(base_path, 'naver_shopping.txt')
    steam_file = os.path.join(base_path, 'steam.txt')
    output_file = os.path.join(base_path, 'preprocessed_data.csv')

    # 기존 파일 삭제
    if os.path.exists(output_file):
        os.remove(output_file)

    files_to_process = [
        (naver_file, 'naver'),
        (steam_file, 'steam')
    ]

    total_processed = 0
    header = True
    seen_reviews = set() # 중복 체크를 위한 집합

    for file_path, dtype in files_to_process:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping...")
            continue
            
        print(f"\nProcessing {dtype} data: {file_path}")
        
        reader = pd.read_csv(file_path, sep='\t', header=None, chunksize=CHUNK_SIZE)
        
        for i, chunk in enumerate(reader):
            # 전처리 수행 (seen_reviews 전달)
            processed_chunk = process_chunk(chunk, seen_reviews, data_type=dtype)
            
            # SAMPLE_SIZE 제한 적용
            if SAMPLE_SIZE is not None:
                remaining = SAMPLE_SIZE - total_processed
                if remaining <= 0:
                    break
                if len(processed_chunk) > remaining:
                    processed_chunk = processed_chunk.iloc[:remaining]
            
            # 결과 저장
            processed_chunk.to_csv(output_file, mode='a', index=False, header=header, encoding='utf-8-sig')
            
            header = False
            total_processed += len(processed_chunk)
            
            print(f"Batch {i+1} processed. Unique count so far: {total_processed}")
            
            del chunk
            del processed_chunk
            gc.collect()

            if SAMPLE_SIZE is not None and total_processed >= SAMPLE_SIZE:
                print(f"Reached SAMPLE_SIZE ({SAMPLE_SIZE}). Stopping...")
                break
        
        if SAMPLE_SIZE and total_processed >= SAMPLE_SIZE:
            break

    print(f"\nPreprocessing complete! Total {total_processed} unique records saved to {output_file}")

if __name__ == "__main__":
    main()
