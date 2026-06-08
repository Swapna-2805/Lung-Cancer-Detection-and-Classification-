# Lung-Cancer-Detection-and-Classification-
Title
Lung Cancer Detection and Classification Using Cat Swarm Optimization and Deep Ensemble Learning Model on CT Images

Description
This project implements a computerized lung cancer detection and classification framework named CSODEL-LCC. The model uses CT scan images to detect and classify lung cancer. The proposed approach combines UNet++ segmentation, Cat Swarm Optimization (CSO) for hyperparameter tuning, fusion-based feature extraction using InceptionResNetV2 and EfficientNetB5, and an ensemble classification model using LSTM, BiLSTM, and GRU. The objective of the system is to improve lung cancer classification accuracy and support early diagnosis using CT image analysis.

Dataset Information
The study uses the LUNA16 Part 1/2 CT image dataset for lung cancer detection and classification. The dataset contains lung CT scan images that are used for preprocessing, segmentation, feature extraction, and classification. The CT images are first enhanced through preprocessing steps such as contrast improvement, noise reduction, and normalization. After preprocessing, lung regions and possible nodules become clearer for automated model analysis.

Dataset Link:
https://www.kaggle.com/datasets/fanbyprinciple/luna-lung-cancer-dataset

Code Information
The implementation code includes modules for CT image loading, preprocessing, UNet++-based segmentation, CSO-based hyperparameter optimization, feature extraction using InceptionResNetV2 and EfficientNetB5, and ensemble classification using LSTM, BiLSTM, and GRU. The RMSProp optimizer is used to optimize the deep learning classifiers. The code also includes performance evaluation using accuracy, precision, recall, specificity, F1-score, MCC, and execution time.

Usage Instructions
To use the project, first download the LUNA16 CT image dataset from the Kaggle link and place the dataset in the project directory. Then install the required Python libraries and run the preprocessing script to resize, normalize, and enhance the CT images. After preprocessing, run the segmentation module to extract lung regions using UNet++. Next, execute the feature extraction module using InceptionResNetV2 and EfficientNetB5. Finally, run the ensemble classification script to classify the CT images as benign or malignant. The final output includes predicted class labels, performance metrics, and graphical results.


Requirements
The following software and Python libraries are required:
Python 3.x
TensorFlow / Keras
NumPy
Pandas
OpenCV
Scikit-learn
Matplotlib
Seaborn
Imbalanced-learn
Jupyter Notebook or Google Colab
CUDA-supported GPU environment is recommended for faster training


Methodology
The proposed CSODEL-LCC method follows a structured pipeline. First, CT scan images are collected from the LUNA16 dataset. Then preprocessing is applied to improve image quality by reducing noise, normalizing intensity values, and enhancing contrast. After preprocessing, UNet++ is used to segment the lung region and isolate the region of interest. The Cat Swarm Optimization algorithm is applied to tune important hyperparameters of the segmentation and learning models. Next, features are extracted using a fusion of InceptionResNetV2 and EfficientNetB5. The extracted features are passed to an ensemble classifier consisting of LSTM, BiLSTM, and GRU models. Finally, the classification performance is evaluated using standard assessment metrics.

Citations

The dataset used in this study is cited as:

LUNA Lung Cancer Dataset:
https://www.kaggle.com/datasets/fanbyprinciple/luna-lung-cancer-dataset
The main manuscript also cites related deep learning, segmentation, optimization, and lung cancer classification studies in the reference section



