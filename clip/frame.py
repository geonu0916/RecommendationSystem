import cv2
import os

def video_to_frames(video_path, frames_dir, every_n_frame=1):
    """
    비디오를 프레임별로 분할하고, 각 프레임을 이미지 파일로 저장합니다.
    
    :param video_path: 비디오 파일의 경로
    :param frames_dir: 프레임 이미지를 저장할 디렉터리의 경로
    :param every_n_frame: 저장할 프레임의 간격 (기본값 1, 모든 프레임을 저장)
    """
    # 디렉터리가 없으면 생성
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 비디오 파일을 열기
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        # 프레임별로 비디오를 읽기
        ret, frame = cap.read()
        # 더 이상 프레임이 없으면 중단
        if not ret:
            break
        # 설정된 간격에 따라 프레임 저장
        if frame_count % every_n_frame == 0:
            frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    # 비디오 파일 닫기
    cap.release()

# 비디오 파일 경로와 프레임을 저장할 디렉터리 설정
video_path = "/home/dm-tomato/dm-gun/metaData/myqt/cctv.mp4" # 비디오 파일 경로를 여기에 입력
frames_dir = "/home/dm-tomato/dm-gun/metaData/anotherModel/clip/video_frame/" # 프레임을 저장할 디렉터리

# 함수 실행
video_to_frames(video_path, frames_dir)
