# chest-xray-classification
Implementing a Convolutional Neural Network model for classification of chest x-ray images

## Summary
* [Introduction & General Information](#introduction--general-information)
* [Objectives](#objectives)
* [Data Used](#data-used)
* [Approach & Methodology](#approach--methodology)
* [Conclusion](#conclusion)
* [Run Locally](#run-locally)


## Introduction & General Information
**Convolutional Neural Networks (CNN)**

- CNN are widely used for image classification. It is a process in which we provide input to the model in the form of images and obtain the image class or the probability that the input image belongs to a particular class.
- Humans can look at an image and recognize what it is, but it is not same for machines. Each image is a series of pixel values arranged in particular order. We have to represent the image in a manner a machine can understand. This can be achieved using a CNN Model.
- If we have a black and white image, the pixels are arranged in the form of 2D array. Each pixel has a value between 0 - 255.
  - 0 means completely white
  - 255 means completely black
  - Grayscale exists if the number lies between 0 and 255
- If we have a colored image, then the pixels are arranged in the form of 3D array. This 3D array has blue, green and red layers.
  - Each color has pixel values ranging from 0 - 255
  - We can find the exact color by combining the pixel values of each of the 3 layers.

**Keras and TensorFlow**
- In deep learning or machine learning, we have datasets which are mostly multi-dimensional, where each dimension represents different features of the data.
- Tensor is the way of representing such multi-dimensional data. In this project, we are dealing with fashion object images. There can be many aspects to an image such as shape, edges, boundaries etc. In order to classify these images correctly as different objects, the convolutional network will have to learn to discriminate these features. These features are incorporated by TensorFlow
- Keras is a high-level library which is built on top of TensorFlow. It provides a scikit-learn type API for building Neural Networks. Hence, it simplifies the process of building neural networks as we don't have to worry about mathematical aspects of TensorFlow algebra.

## Objectives
- Building a convolutional neural networks model using Keras to classify the images of fashion articles into 10 different class items.
- Evaluating the performance of the model using classification report(Precision, Recall, F1-Score and Accuracy) and Confusion matrix.


## Data Used

- We are using NIH X-ray dataset for the purpose of this project. This dataset consists of 112K chest x-ray images. 
  - There are total 784 pixels (28 x 28) in each image.
  - Each pixel has an integer value between the range 0 - 255.
  - Higher integer value represents darker pixel and lower integer value represents lighter pixels.

- Each image is associated with one of the 15 classes (14 Diseases and one for "No-findings"). These classes are namely:
  - 0 - Atelectasis
  - 1 - Consolidation
  - 2 - Infiltration
  - 3 - Pneumothorax
  - 4 - Edema
  - 5 - Emphysema
  - 6 - Fibrosis
  - 7 - Effusion
  - 8 - Pneumonia
  - 9 - Pleural_thickening
  - 10 - Cardiomegaly
  - 11 - Nodule 
  - 12 - Hernia
  - 13 - Mass

## Approach & Methodology
- Loading the chest x-ray dataset and converting the train/test data into pandas dataframe. These dataframes will have class label as the first column followed by 784 columns for pixel values.
- Converting these dataframes into numpy array as it is the acceptable form of input for TensorFlow and Keras.
- Pre-processing train and test numpy arrays in order to make them ready to be fed into the CNN model.
- Visualizing few of the images from the train dataset to get better insight of the data being used.
- Building and training convolutional neural network model based on the training dataset.
- Evaluating the performance of class predictions using test dataset.


## Conclusion
- Model was able to classify the images with 82% accuracy.

## Run Locally
- Make sure Python 3 is installed. Reference to install: [Download and Install Python 3](https://www.python.org/downloads/)
- Clone the project: `git clone https://github.com/setu-parekh/chest-xray-classification.git`
- Route to the cloned project: `cd chest-xray-classification`
- Install necessary packages: `pip install -r requirements.txt`
- Run Jupyter Notebook: `jupyter notebook`
- Select the notebook to open: `chest_xray_classification.ipynb`

