import pandas as pd
import cudf

movies = pd.read_csv('/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/movies_rating.csv')
user = pd.read_csv('/home/dm-tomato/dm-gun/metaData/recommendSystem/processMovie/rating_process.csv')

movies_ratings = pd.DataFrame(movies)
user_ratings = pd.DataFrame(user)
# movieId별로 그룹을 짓고, 각 영화의 평균 평점을 계산
average_ratings = movies_ratings.groupby('movieId')['rating'].mean().reset_index()

# 결과 출력
average_ratings_sorted = average_ratings.sort_values(by='rating', ascending=False)

# 평균 평점이 높은 상위 1000개 영화 출력
top_movies = average_ratings_sorted.head(2000)

filtered_ratings = user_ratings[user_ratings['rating'] >= 3.0]

# movieId별로 그룹을 짓고, 각 영화의 평균 평점을 계산
average_ratings = filtered_ratings.groupby('movieId')['rating'].mean().reset_index()

# 평균 평점이 높은 순으로 정렬
average_ratings_sorted = average_ratings.sort_values(by='rating', ascending=False)

# 평균 평점이 높은 상위 2000개 영화 출력 (데이터셋 크기에 따라 결과 수 조정)
user_movies = average_ratings_sorted.head(2000)

def calculate_precision_recall_f1(user_movies, top_movies, k=1301):
    # 상위 k개 추천된 영화의 movieId 추출
    recommended_at_k_ids = user_movies.head(k)['movieId'].tolist()
    # 상위 2000개 영화의 movieId 추출
    top_movie_ids = top_movies['movieId'].tolist()
    
    a_set = set(top_movie_ids)
    b_set = set(recommended_at_k_ids)
    print(len(a_set), len(b_set))
    
    # 실제 선호하는 아이템과 추천된 아이템의 교집합 개수 (True Positive)
    TP = len(a_set & b_set)
    FP = len(b_set - a_set)
    FN = len(a_set - b_set)
    print(TP, FP, FN)
    
    # Precision@k 계산
    precision_at_k = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall@k 계산
    recall_at_k = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1-score 계산
    if precision_at_k + recall_at_k == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    
    return precision_at_k, recall_at_k, f1_score

# Precision@k, Recall@k, F1-score 계산
precision, recall, f1 = calculate_precision_recall_f1(user_movies, top_movies, k=1000)
print(f"Precision@k: {precision:.4f}, Recall@k: {recall:.4f}, F1-score: {f1:.4f}")