import cv2
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if count % 10 == 0:
            frame_name = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_name, frame)
        
        count += 1
    
    cap.release()
    print("Frames extracted successfully!")

extract_frames("dataset/real_videos/video1.mp4", "extracted_frames/real")