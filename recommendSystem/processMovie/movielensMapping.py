import pandas as pd
from tqdm import tqdm

# 파일 경로 설정
movies_file_path = '/home/dm-tomato/dm-gun/Glocal_K/data/MovieLens_100K/movielens_100k_u1.base'
keywords_file_path = '/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_process_absorption_clip.csv'
output_file_path = '/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_user_clip.csv'

# 키워드 파일 읽기 (헤더가 있다고 가정하고, 첫 번째 줄을 헤더로 사용)
keywords_df = pd.read_csv(keywords_file_path, sep='\t', error_bad_lines=False)

# 영화 파일 읽기
movies_df = pd.read_csv(movies_file_path, sep='\t', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# 새로운 열 생성 및 초기화
movies_df['Keyword'] = None

# 데이터 유형 확인 및 필요한 경우 조정 (예시로 추가, 실제 데이터에 따라 조정 필요)
# movies_df['MovieID'] = movies_df['MovieID'].astype(str)
# keywords_df['movieId'] = keywords_df['movieId'].astype(str)

# ID 기반으로 키워드 매칭 및 업데이트
for index, row in tqdm(movies_df.iterrows(), total=movies_df.shape[0]):
    keyword_row = keywords_df[keywords_df['movieId'] == row['MovieID']] # 영화 ID와 일치하는 키워드 행 찾기
    if not keyword_row.empty:
        movies_df.at[index, 'Keyword'] = keyword_row['keyword'].iloc[0] # 첫 번째 일치하는 키워드 사용

# 변경된 DataFrame을 새로운 CSV 파일로 저장
movies_df.to_csv(output_file_path, sep='\t', index=False)

