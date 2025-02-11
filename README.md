## ** Description and purpose of the project **

Emotion classification model based on facial images. The aim of the project is to create a model that will be based on training sets classified emotions recognized in images of a human face. So, model
will automatically recognize emotions based on facial images using the use of convolutional neural networks (CNN). These networks will analyze such characteristics such as the arrangement of the eyes,
eyebrows or mouth in order to classify facial expressions.Each of the expressions will be classified into one of seven emotion categories:
• Angry (presenting anger)
• Disgust (presenting disgust)
• Fear (presenting fear)
• Happy (presenting happiness)
• Neutral (presenting a neutral emotional state)
• Sad (presenting sadness)
• Surprise (presenting surprise)
Each category can be seen in the attachment below, which presents the separated emotions a set of images in the train set – used to train the model. The submitted images are formatted to 48x48 grayscale pixels,
which is intended focusing the model on key expressive features without color distortion or background.

## ** Metodology **

## Model Architecture Selection

The method chosen for developing the model was a Convolutional Neural Network (CNN) due to its effectiveness in image processing. 
The efficiency of CNN lies in its ability to automatically learn significant visual features, in this case, facial features such as the shape of the eyes, eyebrows, and mouth arrangement.

## Robustness to Image Variations

Furthermore, CNN is resistant to minor changes in the image, meaning that factors such as lighting intensity and small variations in the positioning of key facial features—including slight changes in facial
expressions—will have a lesser impact (Debnath, A., Reza, M. M., Rahman, A., Beheshti, A., Band, S. S., & Alinejad-Rokny, H., 2022). Due to this robustness, emotion classification becomes easier and more
resistant to  distortions.

## Use of MaxPooling Layers

To further reduce the risk of distortions and overfitting, MaxPooling layers were applied (Manduk, 2019). 
These layers enhance model efficiency by reducing the number of parameters, focusing only on the key features of the image.

## Regularization – Dropout

To improve model generalization, the Dropout method was implemented, which randomly sets the activation values of selected neurons to zero during training. 
This limitation forces the network to learn more reliable features rather than relying on the predictive capabilities of a small subset of neurons (Shorten & Khoshgoftaar, 2019).

## Activation Function – Softmax

The model utilizes the Softmax activation function, which is commonly used in multi-class classification tasks. Softmax transforms the raw output values of the network into a probability distribution for each
possible class. In this model, it is an appropriate solution for obtaining probability values for each emotion class (Mamczur, 2021).

## Dataset – FER2013

The FER2013 (Facial Expression Recognition 2013) dataset from Kaggle.com was used to train the model. This dataset is widely utilized for emotion classification tasks. It consists of grayscale images with a
resolution of 48x48 pixels. The images depict faces with various expressions, recorded in a way that ensures the face is centered and occupies a similar area in each image. The training set comprises 28,709 examples, 
while the public test set includes 3,589 examples. ([Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013 )
