import pandas as pd

# 파일 경로 설정
movies_file_path = '/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_titles_movieIds.csv'
keywords_file_path = '/home/dm-tomato/dm-gun/metaData/anotherModel/videoMae/video_classification_results.csv'
output_file_path = '/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_process_absorption_videomae.csv'

# 키워드 파일 읽기
keywords_df = pd.read_csv(keywords_file_path, header=None) # 가정: 키워드 파일에 헤더가 없다고 가정
keywords = keywords_df[0].tolist() # 첫 번째 열의 모든 값을 리스트로 변환

# 영화 제목이 포함된 CSV 파일 읽기
movies_df = pd.read_csv(movies_file_path, sep=',')

# 새로운 열 생성 및 초기화
movies_df['keyword'] = None

# 제목에서 키워드 검색 및 기록
for index, row in movies_df.iterrows():
    title_keywords = [keyword for keyword in keywords if keyword in row['title']]
    if title_keywords:
        movies_df.at[index, 'keyword'] = ', '.join(title_keywords)

# 변경된 DataFrame을 새로운 CSV 파일로 저장
movies_df.to_csv(output_file_path, sep='\t', index=False)