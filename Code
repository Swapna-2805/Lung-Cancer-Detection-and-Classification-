#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# CSODEL-LCC: Lung Cancer Detection and Classification
# Cat Swarm Optimization + Deep Ensemble Learning on CT Images
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import InceptionResNetV2, EfficientNetB5
from tensorflow.keras.utils import to_categorical


# ============================================================
# 1. Dataset Path
# ============================================================

DATASET_PATH = "LUNA16_Dataset/"     # Change this path
IMAGE_SIZE = 224
NUM_CLASSES = 2                      # Benign / Malignant


# ============================================================
# 2. Load CT Images
# ============================================================

def load_ct_images(dataset_path):
    images = []
    labels = []

    class_names = os.listdir(dataset_path)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# ============================================================
# 3. CT Image Preprocessing
# ============================================================

def preprocess_images(images):
    processed = []

    for img in images:
        img = cv2.GaussianBlur(img, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = img / 255.0
        processed.append(img)

    processed = np.array(processed)
    processed = np.expand_dims(processed, axis=-1)

    return processed


# ============================================================
# 4. UNet++ Segmentation Model
# ============================================================

def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    return x


def build_unetpp(input_shape=(224, 224, 1)):
    inputs = Input(input_shape)

    x00 = conv_block(inputs, 32)
    p0 = MaxPooling2D((2, 2))(x00)

    x10 = conv_block(p0, 64)
    p1 = MaxPooling2D((2, 2))(x10)

    x20 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(x20)

    x30 = conv_block(p2, 256)

    u01 = UpSampling2D((2, 2))(x10)
    x01 = conv_block(Concatenate()([x00, u01]), 32)

    u11 = UpSampling2D((2, 2))(x20)
    x11 = conv_block(Concatenate()([x10, u11]), 64)

    u21 = UpSampling2D((2, 2))(x30)
    x21 = conv_block(Concatenate()([x20, u21]), 128)

    u02 = UpSampling2D((2, 2))(x11)
    x02 = conv_block(Concatenate()([x00, x01, u02]), 32)

    u12 = UpSampling2D((2, 2))(x21)
    x12 = conv_block(Concatenate()([x10, x11, u12]), 64)

    u03 = UpSampling2D((2, 2))(x12)
    x03 = conv_block(Concatenate()([x00, x01, x02, u03]), 32)

    output = Conv2D(1, (1, 1), activation="sigmoid")(x03)

    model = Model(inputs, output)
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. Cat Swarm Optimization for Hyperparameter Tuning
# ============================================================

class CatSwarmOptimization:
    def __init__(self, num_cats=10, max_iter=5, dim=3):
        self.num_cats = num_cats
        self.max_iter = max_iter
        self.dim = dim

        self.lower_bound = np.array([0.00001, 16, 0.1])
        self.upper_bound = np.array([0.001, 64, 0.5])

    def fitness(self, params):
        learning_rate = params[0]
        batch_size = int(params[1])
        dropout_rate = params[2]

        score = 1 / (learning_rate + 0.001) + batch_size * 0.01 - dropout_rate
        return score

    def optimize(self):
        cats = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.num_cats, self.dim)
        )

        best_cat = cats[0]
        best_score = self.fitness(best_cat)

        for iteration in range(self.max_iter):
            for i in range(self.num_cats):
                score = self.fitness(cats[i])

                if score > best_score:
                    best_score = score
                    best_cat = cats[i]

                velocity = np.random.rand(self.dim) * (best_cat - cats[i])
                cats[i] = cats[i] + velocity

                cats[i] = np.clip(cats[i], self.lower_bound, self.upper_bound)

        return best_cat


# ============================================================
# 6. Fusion-Based Feature Extraction
# ============================================================

def build_feature_extractor():
    input_layer = Input(shape=(224, 224, 3))

    inception = InceptionResNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=input_layer
    )

    efficient = EfficientNetB5(
        weights="imagenet",
        include_top=False,
        input_tensor=input_layer
    )

    for layer in inception.layers:
        layer.trainable = False

    for layer in efficient.layers:
        layer.trainable = False

    f1 = GlobalAveragePooling2D()(inception.output)
    f2 = GlobalAveragePooling2D()(efficient.output)

    fused = Concatenate()([f1, f2])

    model = Model(inputs=input_layer, outputs=fused)

    return model


def extract_features(images):
    images_rgb = np.repeat(images, 3, axis=-1)

    feature_model = build_feature_extractor()
    features = feature_model.predict(images_rgb, batch_size=8)

    return features


# ============================================================
# 7. Ensemble Classifier: LSTM + BiLSTM + GRU
# ============================================================

def build_ensemble_classifier(input_shape, num_classes, learning_rate=0.0001, dropout_rate=0.3):
    inputs = Input(shape=input_shape)

    lstm_branch = LSTM(64, return_sequences=False)(inputs)
    lstm_branch = Dropout(dropout_rate)(lstm_branch)

    bilstm_branch = Bidirectional(LSTM(64, return_sequences=False))(inputs)
    bilstm_branch = Dropout(dropout_rate)(bilstm_branch)

    gru_branch = GRU(64, return_sequences=False)(inputs)
    gru_branch = Dropout(dropout_rate)(gru_branch)

    merged = Concatenate()([lstm_branch, bilstm_branch, gru_branch])

    dense = Dense(128, activation="relu")(merged)
    dense = Dropout(dropout_rate)(dense)

    output = Dense(num_classes, activation="softmax")(dense)

    model = Model(inputs, output)

    model.compile(
        optimizer=RMSprop(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 8. Performance Evaluation
# ============================================================

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        specificity = 0

    print("Accuracy     :", round(acc * 100, 2))
    print("Precision    :", round(pre * 100, 2))
    print("Recall       :", round(rec * 100, 2))
    print("Specificity  :", round(specificity * 100, 2))
    print("F1-score     :", round(f1 * 100, 2))
    print("MCC          :", round(mcc * 100, 2))


# ============================================================
# 9. Main Execution
# ============================================================

def main():
    print("Loading CT images...")
    images, labels = load_ct_images(DATASET_PATH)

    print("Preprocessing images...")
    images = preprocess_images(images)

    print("Building UNet++ segmentation model...")
    unetpp = build_unetpp()

    print("Applying segmentation...")
    segmented_images = unetpp.predict(images)

    print("Extracting fusion features...")
    features = extract_features(segmented_images)

    print("Applying SMOTE...")
    smote = SMOTE()
    features_balanced, labels_balanced = smote.fit_resample(features, labels)

    features_balanced = np.expand_dims(features_balanced, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features_balanced,
        labels_balanced,
        test_size=0.2,
        random_state=42,
        stratify=labels_balanced
    )

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    print("Optimizing hyperparameters using CSO...")
    cso = CatSwarmOptimization()
    best_params = cso.optimize()

    best_lr = best_params[0]
    best_batch = int(best_params[1])
    best_dropout = best_params[2]

    print("Best Learning Rate:", best_lr)
    print("Best Batch Size:", best_batch)
    print("Best Dropout:", best_dropout)

    print("Training ensemble classifier...")
    model = build_ensemble_classifier(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=NUM_CLASSES,
        learning_rate=best_lr,
        dropout_rate=best_dropout
    )

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=30,
        batch_size=best_batch,
        verbose=1
    )

    print("Evaluating model...")
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    evaluate_model(y_test, y_pred)

    model.save("CSODEL_LCC_Model.h5")

    print("Model saved as CSODEL_LCC_Model.h5")


if __name__ == "__main__":
    main()

