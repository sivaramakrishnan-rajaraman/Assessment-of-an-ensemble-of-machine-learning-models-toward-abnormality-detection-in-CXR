# Assessment-of-an-ensemble-of-machine-learning-models-toward-abnormality-detection-in-chest-radiographs

Respiratory diseases account for a significant proportion of deaths and disabilities across the world. Chest Xray (CXR) analysis remains a common diagnostic imaging modality for confirming intra-thoracic cardio-pulmonary abnormalities. However, the need for expert radiologists to interpret these radiographs is continuing to increase leading to severe backlogs and interpretation delays. These issues can be mitigated by a computer-aided diagnostic (CADx) system to supplement decision-making and improve throughput while preserving and possibly improving the standard-of-care. This work aims to simplify the analysis in a binary triage classification problem that classifies CXRs into normal and abnormal categories. We evaluate the performance of different ensemble strategies that combine predictions of ML classifiers trained with handcrafted/CNN-extracted features toward the current task. We extract HOG and LBP features and train a binary SVM classifier on the extracted features. We use DL models including a custom CNN, pretrained VGG16, and VGG19 [20] to learn hierarchical feature representations from the CXRs. Finally, we combine the predictions of individual base-learners through different ensemble strategies including majority voting, simple averaging, and weighted averaging to observe for a possible improvement in performance. 

## Citation:

Kindly cite this reference associated with this study if you find these codes useful for your research:

### Rajaraman, S., Sornapudi, S., Kohli, M., and Antani, S. Assessment of an ensemble of machine learning models toward abnormality detection in chest radiographs. In Proc. IEEE EMBC 2019, Berlin, Germany, July 23-27, 2019.

# Pre-requisites:

Anaconda3 >=-5.1.0-Windows-x86_64

Jupyter Notebook

Keras >= 2.2.4

Tensorflow-GPU >= 1.9.0

# Code: 

The repository includes the notebook “Ensemble of ML models toward abnormality detection in chest radiographs.ipynb”

# Dataset:

The dataset used in this study is made available by RSNA for the Kaggle pneumonia detection challenge (https://www.kaggle.com/c/rsna-pneumonia-detectionchallenge/data). The dataset includes images with pulmonary opacities that may represent pneumonia and other images that are normal and those without a pulmonary opacity suspicious for pneumonia. 

# Preprocessing:

The lung ROI is segmented using the all-dropout UNET (AD-UNET) to help the base-learners learn relevant information toward arriving at the predictions. After lung segmentation, the resulting images are cropped to the size of a bounding box containing all the lung pixels and resized to 224×224 pixel dimensions for further study. 

# Pretrained DL models:

We used the pretrained VGG16 and VGG19 models and customized them for the task under study. The models are truncated at the deepest convolutional layer, a GAP and dense layer are added to predict on the outcome. The VGG16 model is trained end to end to learn CXR-specific feature representations and categorize them to their respective classes. The VGG19 model is instantiated with the convolutional base and loaded with the pretrained weights. The activation maps before the dense, fully-connected layers are extracted and a dense model is trained on top of the stored features. We performed a randomized grid search to obtain the optimal values for the hyperparameters including learning rate, momentum, and L2-regularization.

# Performance metrics evaluation:

We performed 5-fold cross validation and presented the results in terms of mean and standard deviation. The models’ performance is evaluated in terms of accuracy, the area under the ROC curve (AUC), F-score and Matthews Correlation Coefficient (MCC). We performed multiple ensembles of the predictions of individual base-learners through averaging, majority voting, and weighted averaging strategies to classify the CXRs into normal and abnormal categories. 

# Results:

In weighted averaging, we awarded high/low importance to the predictions by assigning higher weights to more accurate base-learners. We found that the VGG16 outperformed the other methods toward the current task. Thus, we assigned weights of [0.1, 0.4, 0.1, 0.1, 0.3] to the predictions of the custom CNN, VGG16, VGG19, LBP/SVM, and HOG/SVM models respectively. We observed that weighted averaging outperformed majority voting and simple averaging ensembles by achieving an accuracy of 98.7±0.078, AUC of 100±0.02, F-score of 99.1±0.05, and MCC of 96.8±0.18.
