# import cv2
# import numpy as np
# import tensorflow as tf

# IMG_SIZE = 224
# model = tf.keras.models.load_model("model/deepfake_model.h5")

# video_path = "test_video.mp4"  # <-- your video here
# cap = cv2.VideoCapture(video_path)

# frame_number = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_number += 1

#     # Resize
#     img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     # prediction = model.predict(img)[0][0]
    
#     T = 10
#     preds = []

#     for _ in range(T):
#         p = model(img, training=True)
#         preds.append(p.numpy()[0][0])

#     mean_pred = np.mean(preds)
#     variance = np.var(preds)

#     print(f"Frame {frame_number}: Mean={mean_pred}, Var={variance}")

#     # print(f"Frame {frame_number}: {prediction}")

# cap.release()

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 224
T = 10  # Monte Carlo runs
video_path = "test_video.mp4"  # change if needed
threshold = 0.7  # fake probability threshold
fps = 30  # adjust according to your video FPS

# ----------------------------
# LOAD MODEL
# ----------------------------
model = tf.keras.models.load_model("model/deepfake_model.h5")

# ----------------------------
# OPEN VIDEO
# ----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
else:
    print("Video opened successfully.")

frame_number = 0

frame_means = []
frame_variances = []

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Monte Carlo Dropout
    preds = []
    for _ in range(T):
        prediction = model(img, training=True)
        preds.append(prediction.numpy()[0][0])

    mean_pred = np.mean(preds)
    variance = np.var(preds)

    frame_means.append(mean_pred)
    frame_variances.append(variance)

    print(f"Frame {frame_number}: Mean={mean_pred:.4f}, Var={variance:.6f}")

cap.release()

print("Video processing complete.")
print("Total frames processed:", frame_number)

# ----------------------------
# PLOT FAKE PROBABILITY
# ----------------------------
plt.figure()
plt.plot(frame_means)
plt.title("Fake Probability Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Fake Probability")
plt.show()

# ----------------------------
# PLOT UNCERTAINTY
# ----------------------------
plt.figure()
plt.plot(frame_variances)
plt.title("Uncertainty (Variance) Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Predictive Variance")
plt.show()

# ----------------------------
# DETECT FAKE FRAMES
# ----------------------------
fake_frames = []

for i, prob in enumerate(frame_means):
    if prob > threshold:
        fake_frames.append(i)

print("Total Fake Frames Detected:", len(fake_frames))

# ----------------------------
# CONVERT TO TIME
# ----------------------------
fake_times = [frame / fps for frame in fake_frames]

if fake_times:
    print("Suspicious Time (seconds):")
    print(f"From {min(fake_times):.2f}s to {max(fake_times):.2f}s")
else:
    print("No suspicious segments detected.")

# ----------------------------
# SAVE RESULTS TO CSV
# ----------------------------
import pandas as pd

results = pd.DataFrame({
    "Frame": list(range(len(frame_means))),
    "Fake_Probability": frame_means,
    "Uncertainty": frame_variances
})

results.to_csv("video_analysis_results.csv", index=False)

print("Results saved to video_analysis_results.csv")