import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, brier_score_loss

IMG_SIZE = 224
BATCH_SIZE = 32
T = 20  # number of MC runs

# Load model
model = tf.keras.models.load_model("model/deepfake_model.h5")

# Test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

true_labels = test_data.classes

mc_predictions = []

print("Running Monte Carlo Dropout...")

for i in range(len(test_data)):
    img, _ = test_data[i]
    preds = []

    for _ in range(T):
        prediction = model(img, training=True)
        preds.append(prediction.numpy()[0][0])

    mc_predictions.append(preds)

mc_predictions = np.array(mc_predictions)

mean_preds = mc_predictions.mean(axis=1)
variance_preds = mc_predictions.var(axis=1)

# Metrics
auc = roc_auc_score(true_labels, mean_preds)
brier = brier_score_loss(true_labels, mean_preds)

print("ROC-AUC:", auc)
print("Brier Score:", brier)

print("Average Predictive Variance:", np.mean(variance_preds))