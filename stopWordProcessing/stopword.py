import json
import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

file_path = '/home/dm-tomato/dm-gun/metaData/file/1 Five Thief Caught on CCTV camera.json'
save_path = '/home/dm-tomato/dm-gun/metaData/processFile/1 Five Thief Caught on CCTV camera_processed.csv'  # 확장자를 .csv로 변경

# JSON 파일 로드
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 불용어 목록 로드
stop_words = set(stopwords.words('english'))

# 탐지된 객체의 이름 추출
detected_objs = []
for frame_key, frame_value in data.items():
    if frame_key.startswith('Frame'):
        detected_objs.extend(frame_value['DetectedObjsDict'].keys())

# 객체 이름 토큰화 및 불용어 제거
word_tokens = word_tokenize(' '.join(detected_objs))
filtered_words = [word for word in word_tokens if word not in stop_words]

# 결과를 CSV 파일로 저장
with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # CSV 파일 헤더 작성
    csvwriter.writerow(['Filtered Words'])
    # 불용어 제거된 단어 목록을 행별로 저장
    for word in filtered_words:
        csvwriter.writerow([word])

print(f'불용어 제거 후 결과가 {save_path}에 저장되었습니다.')
