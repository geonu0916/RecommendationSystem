import pandas as pd

# 파일 경로 설정
csv_file_path = '/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_process_title.csv'
output_file_path = '/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_process_title2.csv'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)

# 모든 셀에서 쉼표(,) 제거
# `applymap` 함수는 DataFrame의 각 셀에 함수를 적용합니다.
df = df.applymap(lambda x: x.replace(',', '') if isinstance(x, str) else x)

# 변경된 데이터를 새 파일로 저장
df.to_csv(output_file_path, index=False)