from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch
import cv2
import numpy as np

# 비디오 파일 경로
video_path = '/home/dm-tomato/dm-gun/metaData/myqt/cctv.mp4'

# 비디오에서 프레임을 추출하는 함수
def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV에서 읽은 프레임은 BGR 형식이므로 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    # 프레임이 모델에 요구하는 크기가 아니면 조정
    frames_resized = [cv2.resize(frame, (224, 224), 
                                 interpolation=cv2.INTER_LINEAR) for frame in frames]
    return np.array(frames_resized, dtype=np.uint8)  # dtype을 명시적으로 지정

# 프레임 추출 및 리사이징
frames = extract_frames(video_path)
print(frames.shape)
# 프로세서 및 모델 초기화
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# 비디오 데이터 처리
inputs = processor(frames, return_tensors="pt")

# 모델 예측
model.eval()  # 평가 모드로 설정
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 예측된 클래스 인덱스 출력
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
