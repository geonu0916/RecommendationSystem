# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_jaccard_similarity_inclusive(file_path1, column1, file_path2, column2):
    # 첫 번째 파일 로드
    df1 = pd.read_csv(file_path1)
    # 두 번째 파일 로드
    df2 = pd.read_csv(file_path2)

    # 지정된 열의 값들을 소문자로 변환하여 집합 생성
    set1 = set(df1[column1].dropna().str.lower().unique())
    set2 = set(df2[column2].dropna().str.lower().unique())

    # '포함' 관계를 고려하여 교집합과 합집합을 구성
    intersection = {s1 for s1 in set1 for s2 in set2 if s1 in s2 or s2 in s1}
    union = set1.union(set2)

    # 수정된 자카드 유사도 계산
    jaccard_similarity_score = len(intersection) / len(union) if union else 0

    return jaccard_similarity_score

# 파일 경로와 열 이름 설정
file_path1 = "/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_process_title2.csv"
column1 = "title"
file_path2 = "/home/dm-tomato/dm-gun/metaData/keyword_extraction/TextRank/keyword/extracted_1 Five Thief Caught on CCTV camera2.csv"
column2 = "Keyword"

def create_similarity_matrix_and_plot(file_path1, column1, file_path2, column2):
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    
    items1 = df1[column1].dropna().str.lower().unique()
    items2 = df2[column2].dropna().str.lower().unique()
    
    similarity_matrix = pd.DataFrame(index=items1, columns=items2, dtype=float)
    
    for i in items1:
        for j in items2:
            similarity_matrix.at[i, j] = int(i in j or j in i)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm')
    plt.title("Mapping Heatmap")
    plt.show()

# 자카드 유사도 계산
similarity_score = calculate_jaccard_similarity_inclusive(file_path1, column1, file_path2, column2)
print(f"Jaccard Similarity Score: {similarity_score}")
create_similarity_matrix_and_plot(file_path1, column1, file_path2, column2)
# %%
