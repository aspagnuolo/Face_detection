# Face Detection from scratch: Comprehensive Framework and Analysis

This repository provides scripts and resources for creating a custom face detection system without using pre-trained models. The approach leverages *Histogram of Oriented Gradients* (HOG) features for object detection, along with data augmentation, sliding windows, non-maximum suppression, and Support Vector Machines (SVM) classification.

## Contents

- `neg_dataset/`: Folder containing negative image samples.
- `pos_dataset/`: Folder containing positive image samples.
- `test/`: Folder containing test images for evaluation.
- `Face_Detection.ipynb`: Jupyter Notebook for performing face detection on test images using SVM and hard negatives.
- `train_model.ipynb`: Jupyter Notebook for training the SVM model with augmented positive images, hard negative mining and custom techniques.
- `svm_with_hard_negatives`: Pre-trained SVM model saved using joblib.
- `utils.py`: Utility functions and libraries necessary for script execution.

## Approach Overview

1. **Data Loading and Augmentation**:
   - Positive and negative image samples are loaded from the `pos_dataset` and `neg_dataset` folders, respectively.
   - Positive images are augmented by simulating distance variation and adding mirrored images to increase training data.
   - Negative images are augmented by extracting random patches to diversify the negative dataset.

2. **Feature Extraction and SVM Training**:
   - HOG features are extracted from both positive and augmented negative images.
   - An SVM classifier is trained using the positive and augmented negative features.
   - The SVM model is further enhanced with hard negative mining using the sliding window approach.

3. **Object Detection**:
   - The trained SVM model is used for detecting faces in test images.
   - A sliding window approach is employed to generate image patches for detection.
   - Non-maximum suppression is applied to remove redundant bounding boxes and improve detection accuracy.

## References

•	H. Chandra, ‘Pipelines & Custom Transformers in scikit-learn: The step-by-step guide (with Python code)’, Medium, Jun. 27, 2020. https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156.

•	M. Tyagi, ‘HOG(Histogram of Oriented Gradients)’, Medium, Jul. 24, 2021. https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f

•	N. Dalal and B. Triggs, ‘Histograms of oriented gradients for human detection’, in 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), Ieee, 2005, pp. 886–893.

•	SkImage Documentation : https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=hog#skimage.feature.hog

•	C.-H. Yuan, ‘Face-Detection-with-a-Sliding-Window’. May 30, 2023. [Online]. Available: https://github.com/lionelmessi6410/Face-Detection-with-a-Sliding-Window

•	Rosebrock, ‘Sliding Windows for Object Detection with Python and OpenCV’, PyImageSearch, Mar. 23, 2015. https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

•	‘Face Detection Project’. https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj5/html/agartia3/index.html

•	J. Hosang, R. Benenson, and B. Schiele, ‘Learning non-maximum suppression’, in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 4507–4515.

•	P. S. MAR 8 and 2023 4 Min Read, ‘How to code Non-Maximum Suppression (NMS) in plain NumPy’, Roboflow Blog, Mar. 08, 2023. https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/ 

•	"ProfAI: A Virtual Data Science Coach," ProfessionAI, 2023. Online. Available: https://prof.profession.ai/

•	“ChatGPT”. OpenAI's ChatGPT, 2023 - Conversational AI Language Model. [Online]. Available: https://chat.openai.com/
