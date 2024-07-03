import pandas as pd
import csv

# 원본 CSV 파일 경로
input_csv_path = '/home/dm-tomato/dm-gun/metaData/movielens/movies.csv'
genres = []

with open(input_csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 'title' 컬럼 데이터만 추출하여 리스트에 추가
        genres.append(row['genres'])

df = pd.DataFrame(data)

# 장르를 '|' 기준으로 분리하고 중복을 제거한 후 모든 유니크 장르를 추출
unique_genres = set("|".join(df['genres']).split("|"))

# 결과 출력
unique_genres
