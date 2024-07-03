from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import glob
import pandas as pd
from tqdm import tqdm  # tqdm 추가

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 이미지가 저장된 폴더의 경로
images_dir = "/home/dm-tomato/dm-gun/metaData/anotherModel/clip/video_frame"

# 해당 디렉터리 내의 모든 이미지 파일 경로를 가져옴
image_files = glob.glob(f"{images_dir}/*.jpg")  # 확장자에 따라 변경 가능

# 텍스트 프롬프트
texts = ["person", "motorcycle", "bench", "potted plant", "light", "traffic light", "bird", "car", "handbag", "bottle", "skateboard", "bicycle", "skis", "surfboard", "chair", "dog"]

results = []

# 각 이미지에 대해 CLIP 모델 실행, tqdm으로 감싸서 진행 상황 표시
for image_path in tqdm(image_files, desc="Processing images"):
    image = Image.open(image_path)  # 이미지 파일 열기

    # 이미지와 텍스트를 CLIP 프로세서에 입력
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to('cuda')

    # 모델 실행
    outputs = model(**inputs)

    # 이미지-텍스트 유사성 점수
    logits_per_image = outputs.logits_per_image

    # 라벨 확률 계산
    probs = logits_per_image.softmax(dim=1)

    for i, text in enumerate(texts):
        prob = probs[0][i].item()
        if prob >= 0.1:  # 확률 임계값을 조정할 수 있습니다.
            results.append({"image_path": image_path, "label": text, "probability": prob})

# 결과를 DataFrame으로 변환 후 CSV 파일로 저장
df = pd.DataFrame(results)
df.to_csv("image_classification_results.csv", index=False)
