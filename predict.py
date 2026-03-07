import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/deepfake_model.h5")

# Load image
img_path = "test.jpg"  # change this to your image path
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.reshape(img, [1, 224, 224, 3])

# Prediction
# prediction = model.predict(img)

# if prediction[0][0] > 0.5:
#     print("Fake Image")
# else:
#     print("Real Image")
# Prediction
prediction = model.predict(img)

print("Prediction Value:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Fake Image")
else:
    print("Real Image")