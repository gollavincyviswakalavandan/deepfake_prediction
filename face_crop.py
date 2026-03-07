import cv2
import os

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

input_folder = "extracted_frames/real"
output_folder = "extracted_frames/real_faces"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, face_resized)

print("Face cropping completed!")