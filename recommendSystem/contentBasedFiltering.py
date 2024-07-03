import cudf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from scipy.sparse import vstack
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def batch_cosine_similarity(matrix, batch_size):
    # 결과를 저장할 행렬 초기화
    cosine_similarity_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    
    # tqdm을 사용하여 첫 번째 for 반복문에 진행률 표시줄 추가
    for start_row in tqdm(range(0, matrix.shape[0], batch_size), desc="Calculating cosine similarity"):
        end_row = min(start_row + batch_size, matrix.shape[0])
        batch_matrix = matrix[start_row:end_row]
        
        for start_col in range(0, matrix.shape[0], batch_size):
            end_col = min(start_col + batch_size, matrix.shape[0])
            batch_matrix_col = matrix[start_col:end_col]
            
            # 배치에 대한 코사인 유사도 계산
            batch_cosine_sim = linear_kernel(batch_matrix, batch_matrix_col)
            
            # 계산된 코사인 유사도를 결과 행렬에 저장
            cosine_similarity_matrix[start_row:end_row, start_col:end_col] = batch_cosine_sim
            
    return cosine_similarity_matrix

# 데이터 로드
movies = cudf.read_csv('/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_process_absorption.csv',
                       sep="\t", names=['movieId', 'title', 'keywords'], header=None)
genres = cudf.read_csv('/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_process_genres.csv',
                       sep="\t", names=['genres'], header=None)
ratings = cudf.read_csv('/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/rating_process_sort.csv',
                        sep=',', names=['userId', 'movieId', 'rating'], header=None)

# 숫자로 변환할 수 없는 값을 확인
invalid_values = ratings[~ratings['rating'].str.match(r'^-?\d+(?:\.\d+)?$')]
ratings = ratings[ratings['rating'].str.match(r'^-?\d+(?:\.\d+)?$')]
ratings['rating'] = ratings['rating'].astype(float)

# 'description' 컬럼 생성
movies['description'] = genres['genres'].str.cat(movies['title'], sep=" ")

# 텍스트 데이터 전처리
description_pd = movies['description'].to_pandas()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(description_pd)

# 코사인 유사도 계산
#cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# 수정된 함수를 사용하여 코사인 유사도 계산
cosine_sim = batch_cosine_similarity(tfidf_matrix, batch_size=200)

# 주어진 영화의 타이틀로부터 해당 영화의 인덱스를 반환하는 함수
# def get_index_from_title(title):
#     return movies[movies['title'] == title].index[0]

def get_index_from_title(title):
    # 데이터프레임에서 영화 제목이 주어진 제목을 포함하는지 검색 (대소문자 구분 없음)
    matches = movies[movies['title'].str.lower().str.contains(title.lower())]
    if len(matches) == 0:
        print(f"Error: No movies found containing '{title}' in titles.")
        return None
    return matches.index[0]

# 추천된 영화 목록을 가져오는 함수
def recommend_movies(title, cosine_sim=cosine_sim):
    index = get_index_from_title(title)
    if index is None:
        return cudf.DataFrame()
    
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:30001]]
    recommended_titles = movies.iloc[movie_indices][['movieId', 'title']]
    recommended_titles.reset_index(drop=True, inplace=True)
    
    # 평점 정보가 있는 movieId만 필터링
    recommended_movieIds = recommended_titles['movieId'].to_arrow().to_pylist()[1:30002]
    recommended_ratings = ratings[ratings['movieId'].isin(recommended_movieIds) 
                                  & (ratings['rating'] >= 3.0)]
    
    # movieId 별 평균 평점 계산
    average_ratings = recommended_ratings.groupby('movieId').mean().reset_index()
    
    # 평균 평점 정보를 추천된 영화 목록에 결합
    final_recommendations = recommended_titles.merge(average_ratings, on='movieId', how='left')
    final_recommendations = final_recommendations.sort_values(by='rating', 
                                                              ascending=False).reset_index(drop=True)
    
    return final_recommendations

recommendations = recommend_movies('person')

# 추천된 영화의 movieId 목록 추출
recommended_movieIds = [int(x) for x in recommendations['movieId'].to_arrow().to_pylist()]
# 추천된 영화에 대한 사용자 평점 찾기
recommended_ratings = ratings[ratings['movieId'].astype(int).isin(recommended_movieIds)][1:10001]
# 사용자가 실제로 선호하는 영화 목록 (예: 사용자가 평점 3점 이상을 준 영화)
relevant_movieIds = [int(x) for x in ratings[ratings['rating'] >= 3]
                     ['movieId'].to_arrow().to_pylist()][1:2001]
# 평점 정보가 있는 영화에 대해서만 평균 평점 계산
if not recommended_ratings.empty:
    average_rating = recommended_ratings['rating'].mean()
    print(f"추천된 영화의 평균 평점: {average_rating}")
else:
    print("추천된 영화 중 평점 정보가 있는 영화가 없습니다.")

print(recommended_movieIds)
def calculate_precision_recall_f1(recommended_ratings, relevant_movieIds, k=1301):
    recommended_at_k = recommended_ratings[1:k]
    global b_set
    a_set = set(relevant_movieIds)
    b_set = set(recommended_at_k)
    print(len(a_set), len(b_set))
    # 실제 선호하는 아이템과 추천된 아이템의 교집합 개수 (True Positive)
    #TP = len(set(recommended_at_k) & set(relevant_movieIds))
    TP = len(a_set & b_set)
    FP = len(b_set - a_set)
    FN = len(a_set - b_set)
    print(TP, FP, FN)
    
    # Precision@k 계산
    #precision_at_k = TP / float(k)
    precision_at_k = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall@k 계산
    #recall_at_k = TP / float(len(relevant_movieIds))
    recall_at_k = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1-score 계산
    if precision_at_k + recall_at_k == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    
    return precision_at_k, recall_at_k, f1_score

# Precision@k, Recall@k, F1-score 계산
precision, recall, f1 = calculate_precision_recall_f1(recommended_movieIds, relevant_movieIds, k=1001)
print(f"Precision@k: {precision:.4f}, Recall@k: {recall:.4f}, F1-score: {f1:.4f}")