import json
import numpy as np
import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# JSON 파일 읽기
json_file_path = '/home/dm-tomato/dm-gun/metaData/file/1.json'  # JSON 파일 경로 지정
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    # 'obj_to_detect' 리스트를 기반으로 문서(doc) 생성
    doc = ". ".join(data['Metadata']['obj_to_detect'])  # 리스트를 문자열로 변환

# 3개의 단어 묶음인 단어구 추출
n_gram_range = (3, 3)
stop_words = "english"

count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
candidates = count.get_feature_names_out()

print('trigram 개수 :',len(candidates))
print('trigram 다섯개만 출력 :',candidates[:5])

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)