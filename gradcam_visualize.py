# ==========================================
# gradcam_visualize.py
# Fully Corrected Grad-CAM for Sequential Model
# ==========================================

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224
image_path = "test.jpg"   # change this
model_path = "model/deepfake_model.h5"

# ----------------------------
# LOAD MODEL
# ----------------------------
model = tf.keras.models.load_model(model_path)

# Force build model (important for Keras 3)
model.build((None, 224, 224, 3))

print("Model Loaded Successfully")
model.summary()

# ----------------------------
# FIND LAST CONV LAYER
# ----------------------------
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError("No Conv2D layer found in model.")

print("Using Conv Layer:", last_conv_layer.name)

# ----------------------------
# LOAD IMAGE
# ----------------------------
img = cv2.imread(image_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_norm = img / 255.0
img_array = np.expand_dims(img_norm, axis=0)

# ----------------------------
# CREATE GRAD MODEL
# ----------------------------
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.outputs[0]]
)

# ----------------------------
# GRAD-CAM COMPUTATION
# ----------------------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = tf.reduce_mean(predictions)

grads = tape.gradient(loss, conv_outputs)

if grads is None:
    raise ValueError("Gradients are None. Model architecture not compatible.")

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap) + 1e-10
heatmap = heatmap.numpy()

# ----------------------------
# HEATMAP PROCESSING
# ----------------------------
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = heatmap * 0.4 + img

# ----------------------------
# SAVE OUTPUT
# ----------------------------
os.makedirs("gradcam_output", exist_ok=True)
output_path = "gradcam_output/gradcam_result.jpg"
cv2.imwrite(output_path, overlay)

print("Grad-CAM image saved at:", output_path)

# ----------------------------
# DISPLAY RESULT
# ----------------------------
plt.imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Visualization")
plt.axis("off")
# plt.show()