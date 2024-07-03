#%%
import json
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import normalize
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = '/home/dm-tomato/dm-gun/metaData/processFile/1 Five Thief Caught on CCTV camera_processed.csv'

def extract_data_from_csv(csv_file_path):
    objects = []
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            object_name = row['Filtered Words']
            objects.append(object_name)
    return objects

# CSV 파일에서 객체 데이터 읽기
objects = extract_data_from_csv(csv_file_path)

# 단어 빈도 및 인덱스 스캔
def scan_vocabulary(objects, min_count=2):
    counter = Counter(objects)
    counter = {w: c for w, c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}
    print("단어 빈도:", list(counter.items())[:5])  # 처음 5개 항목 출력
    return idx_to_vocab, vocab_to_idx

# 공기 행렬 생성
def cooccurrence(objects, vocab_to_idx, window=2, min_cooccurrence=2):
    counter = defaultdict(int)
    for i, obj in enumerate(objects):
        if obj in vocab_to_idx:
            start = max(0, i - window)
            end = min(len(objects), i + window + 1)
            for j in range(start, end):
                if i != j and objects[j] in vocab_to_idx:
                    counter[(vocab_to_idx[obj], vocab_to_idx[objects[j]])] += 1
                    counter[(vocab_to_idx[objects[j]], vocab_to_idx[obj])] += 1
    print("공기 행렬 카운터:", list(counter.items())[:5])  # 처음 5개 공기 데이터 출력
    n_vocabs = len(vocab_to_idx)
    data = list(counter.values())
    rows, cols = zip(*counter.keys())
    return csr_matrix((data, (rows, cols)), shape=(n_vocabs, n_vocabs))

# 페이지랭크 알고리즘
def pagerank(x, df=0.85, max_iter=30):
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1,1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
    for _ in range(max_iter):
        R = df * (A @ R) + bias
    print("페이지랭크 점수:", R.flatten()[:5])  # 처음 5개 점수 출력
    return R

# 텍스트랭크 키워드 추출
def textrank_keyword(objects, min_count=2, window=2, min_cooccurrence=2, 
                     df=0.85, max_iter=30, topk=30):
    idx_to_vocab, vocab_to_idx = scan_vocabulary(objects, min_count)
    g = cooccurrence(objects, vocab_to_idx, window, min_cooccurrence)
    R = pagerank(g, df, max_iter).reshape(-1)
    idxs = R.argsort()[-topk:]
    keywords = [(idx_to_vocab[idx], R[idx]) for idx in reversed(idxs)]
    print("키워드:", keywords[:5])  # 처음 5개 키워드 출력
    return keywords

# 키워드 추출
keywords = textrank_keyword(objects, min_count=2, window=2, min_cooccurrence=2, 
                            df=0.85, max_iter=30, topk=30)

# JSON 파일로 키워드 저장
def save_keywords_to_json(keywords, output_path, file_name):
    keywords_only = [word for word, _ in keywords]
    full_output_path = f"{output_path}/{file_name}.json"
    with open(full_output_path, 'w', encoding='utf-8') as f:
        json.dump(keywords_only, f, ensure_ascii=False, indent=4)
    print("저장된 키워드 파일 경로:", full_output_path)  # 파일 경로 출력
    return full_output_path

def save_keywords_to_csv(keywords, output_path, file_name):
    # 키워드에서 점수 정보를 제외하고 키워드만 추출
    keywords_only = [word for word, _ in keywords]
    full_output_path = f"{output_path}/{file_name}.csv"
    
    with open(full_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 헤더 추가 (선택사항)
        writer.writerow(['Keyword'])
        # 각 키워드를 별도의 행으로 저장
        for keyword in keywords_only:
            writer.writerow([keyword])
    
    print("저장된 키워드 CSV 파일 경로:", full_output_path)
    return full_output_path

def plot_cooccurrence_matrix(co_matrix, vocab):
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_matrix.toarray(), annot=True, fmt="d", xticklabels=vocab, yticklabels=vocab)
    plt.title("Co-occurrence Matrix Heatmap")
    plt.show()

# 공기 행렬과 어휘 목록을 사용하여 히트맵 생성
idx_to_vocab, vocab_to_idx = scan_vocabulary(objects)
co_matrix = cooccurrence(objects, vocab_to_idx)
plot_cooccurrence_matrix(co_matrix, idx_to_vocab)

def plot_pagerank_scores(R, vocab):
    plt.figure(figsize=(12, 6))
    scores = R.flatten()
    plt.bar(range(len(scores)), scores, tick_label=vocab)
    plt.xticks(rotation=90)
    plt.title("PageRank Scores of Keywords")
    plt.show()

# 페이지랭크 점수와 어휘 목록을 사용하여 바 차트 생성
R = pagerank(co_matrix)
plot_pagerank_scores(R, idx_to_vocab)

# 파일 저장 실행
output_path = '/home/dm-tomato/dm-gun/metaData/keyword_extraction/TextRank/keyword'
file_name = 'extracted_1 Five Thief Caught on CCTV camera2'
full_output_path = save_keywords_to_csv(keywords, output_path, file_name)
# %%
