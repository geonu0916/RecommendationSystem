import json
from sklearn.feature_extraction.text import TfidfVectorizer

# JSON 파일 읽기
json_file_path = '/home/dm-tomato/dm-gun/metaData/file/1 Five Thief Caught on CCTV camera.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 모든 프레임에서 "class" 값을 수집하여 문서 생성
classes = []
for frame_key in data.keys():
    if frame_key.startswith("Frame"):
        frame_data = data[frame_key]
        for obj_key in frame_data["Detail"].keys():
            obj_data = frame_data["Detail"][obj_key]
            classes.append(obj_data["class"])

# 클래스 목록을 공백으로 구분된 하나의 문자열로 변환
corpus = [" ".join(classes)]

# TfidfVectorizer 초기화 및 fit_transform 수행
tfidfv = TfidfVectorizer()
tfidf_matrix = tfidfv.fit_transform(corpus)
feature_names = tfidfv.get_feature_names_out()

# 각 단어의 TF-IDF 점수 얻기
scores = tfidf_matrix.toarray().flatten()

# 단어와 점수를 튜플로 묶어 리스트 생성
words_scores = list(zip(feature_names, scores))

# TF-IDF 점수에 따라 정렬하고 상위 5개 선택
top5_words_scores = sorted(words_scores, key=lambda x: x[1], reverse=True)[:5]

# 상위 5개 단어와 점수 출력
print("Top 5 TF-IDF Words and Scores:")
for word, score in top5_words_scores:
    print(f"{word}: {score:.4f}")