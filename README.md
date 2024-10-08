#Panorex.AI - University Graduation Project
Panorex.AI is an innovative web platform designed to give users insights into their oral and dental health by evaluating dental tomography images with the power of artificial intelligence. The platform name, "Panorex.AI," is derived from the combination of panoramic X-rays and AI (artificial intelligence), reflecting its core functionality.

###Project Overview
Panorex.AI empowers users to upload their dental X-rays (panoramic radiographs) and receive AI-supported analyses. By leveraging a state-of-the-art machine learning model, the platform assists patients in better understanding their dental health and potential anomalies from the comfort of their home.

###Objective
This project aims to analyze dental tomography images, detect anomalies, and present findings in a clear and user-friendly format. The core technology stack includes:

Python for data handling and preprocessing,
PyTorch for model training and neural network implementation,
Flask as the backend web framework to deliver a seamless user interface.

###Machine Learning Model
We employed Faster R-CNN, a leading object detection algorithm, to identify specific dental conditions in X-ray images. The model was trained and tested on a meticulously prepared dataset, labeled in collaboration with experienced dentists. Data augmentation techniques, such as rotation, reflection, and image addition, were applied to enhance model robustness and improve detection accuracy.

###Dataset
Our dataset consists of 600 training images and 50 test images, annotated using the LabelImg tool to mark relevant dental structures and anomalies. Collaboration with professionals ensures precise labeling, which is crucial for model performance. The preprocessing steps included:

Image normalization,
Data augmentation (rotation, reflection, and addition).

###Key Features
AI-Driven Analysis: Get immediate insights into your dental health through AI-based tomography analysis.
User-Friendly Interface: Easily upload dental X-rays and view results in a simple, intuitive layout.
Collaboration with Experts: Datasets were developed in partnership with dentists to ensure high-quality and clinically relevant annotations.

###Performance & Results
We conducted multiple experiments to assess the performance of our model. Key evaluation metrics include:

Accuracy, Precision, Recall, and F1-Score. Through cross-validation, we obtained an initial accuracy of 60%, identifying potential areas for further optimization and refinement.

###Future Work
Our current focus is on improving model accuracy and generalization by expanding the dataset and exploring more sophisticated augmentation techniques. We are also working towards integrating additional features, such as a detailed dental health report and real-time X-ray analysis capabilities.
