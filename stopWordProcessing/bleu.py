from nltk.translate.bleu_score import sentence_bleu

# 참조 문장과 후보 문장 정의
reference = [["Ne", "le", "dis", "à", "personne"]]
candidate = ["personne"]

# 1-gram BLEU 점수
score_1gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(f"1-gram BLEU Score: {score_1gram}")

# 2-gram BLEU 점수
score_2gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
print(f"2-gram BLEU Score: {score_2gram}")

# 3-gram BLEU 점수
score_3gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
print(f"3-gram BLEU Score: {score_3gram}")

# 4-gram BLEU 점수
score_4gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(f"4-gram BLEU Score: {score_4gram}")
