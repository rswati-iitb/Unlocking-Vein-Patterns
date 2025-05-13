# -*- coding: utf-8 -*-
"""
Created on Sun May 11 19:34:20 2025

@author: 91887
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

img = cv2.imread('palmvein.png')
# Constants for LBP
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# Placeholder for BSIF filters (normally learned offline or loaded)
def dummy_bsif(img):
    # Simulate BSIF by applying a set of Gabor-like filters and thresholding
    kernel = cv2.getGaborKernel((5, 5), 1.0, np.pi/4, 10.0, 0.5)
    filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    _, binary = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    return binary

# Simulated MRF-based smoothing
def mrf_smoothing(img):
    return cv2.GaussianBlur(img, (3, 3), 0)

# Load sample image and preprocess
def load_and_preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    return img

# Extract LBP features
def extract_lbp_features(img):
    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    return hist / np.sum(hist)

# Extract CNN features using VGG16 (decision-level features)
def extract_cnn_features(img):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_array = img_to_array(img_color)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    features = model.predict(img_array)
    return features.flatten()

# Combine all features
def extract_combined_features(img):
    lbp_feat = extract_lbp_features(img)
    mrf_img = mrf_smoothing(img)
    bsif_feat = dummy_bsif(mrf_img).flatten()[:256] / 255.0
    cnn_feat = extract_cnn_features(img)[:512]
    return np.concatenate([lbp_feat, bsif_feat, cnn_feat])

# Classification pipeline (dummy for illustration)
def classify_image(image_path):
    img = load_and_preprocess_image(image_path)
    features = extract_combined_features(img)
    features = StandardScaler().fit_transform([features])
    model = SVC()
    # Normally: model.fit(X_train, y_train)
    # Here we just simulate a prediction
    prediction = model.predict(features) if hasattr(model, "predict") else "Simulated_Class"
    return prediction

# Example usage
# prediction = classify_image("example.jpg")
# print("Predicted class:", prediction)
