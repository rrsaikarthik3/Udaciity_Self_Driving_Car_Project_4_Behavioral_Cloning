# **Traffic Sign Recognition** 

## Writeup



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Hysteresis_plot.jpg "Visualization"
[image2]: ./examples/Pre-process_ex.jpg "Initial Image before pre-processing"
[image3]: ./examples/Post_Augmentation_Hysteresis_plot.jpg "Post Augmentation Visualization"
[image4]: ./examples/Rotated_Image.jpg "Rotated Image"
[image5]: ./new_test_images/children-crossing.jpg "Children Crossing"
[image6]: ./new_test_images/dang_right.jpg "Dangerous Right"
[image7]: ./new_test_images/dang_left.jpg "Dangerous Left"
[image8]: ./new_test_images/gen_Caution.jpg "General Caution"
[image9]: ./new_test_images/go_straight_or_right.jpg "Go Straight or Right"
[image10]: ./new_test_images/keep_left.jpg "Keep Left"
[image11]: ./new_test_images/road_work.jpg "Road Work"
[image12]: ./new_test_images/road_work_1.jpg "Keep Left"
[image13]: ./new_test_images/slippery_road.jpg "Slippery Road"
[image14]: ./new_test_images/slippery_road_1.jpg "Slippery Road"


## Rubric Points

### Writeup / README


The link to the project is [project code](https://github.com/rrsaikarthik3/Project_3_Classifying_Traffic_signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. The training, validation and test datasets were loaded and the basic information about them were collected using the standard Python functions and Numpy libraries


* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Visaulization of Dataset trend

Here is an exploratory visualization of the data set. It is a hysteresis plot showing number of training data points available for each class

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Data Augmentation and Pre-Processing
1. Beofre pre-processing the data, the trianing data size was increased by rotating the existing images by 4 different angles about the center of the image. Initially without the additional data, I could achive the validation accuracy of over 93%. But, this had very poor results on the completely new dataset I downloaded from the Internet. I tried using different approaches to make the model better by grayscaling it and adding it as a layer in addition to the existing RGB layer and this did not have much impact. I even tried removing the RGB layers and just trained with the grayscale images , even though this slightly improved the performance on the validation set, it still did not improve the performance on the completely new dataset. This motivated me to increase the training samples. 
A sample of the original pre-processed image is 


![alt text][image2]


2. Thus, the training data was made to 5x the initial size. 
The visualization of the dataset post augmentation is shown here:


![alt text][image3]


A sample of the rotated image is shown here: 


![alt text][image4]


3. Then, the images were normalized before training
Normalized Image sample is shown here:






#### 2. The Training Architecture

The architecture used was the LeNet architecture. The main modifications done were in the height of each layer and the convolution size in the convolution layer.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x25 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,Same Padding,  outputs 14x14x25 				|
| Convolution 5x5	    |  1x1 stride, Valid padding, outputs 10x10x50     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,Same Padding,  outputs 5x5x50 				|
| Fully connected		| 1250 -> 400        									|
| RELU					|												|
| Dropout				|	 Rate = 0.5											|
| Fully connected		| 400 -> 100        									|
| Fully connected		| 100 -> 43        									|

 


#### 3. Training Model
To train the model, I used the Adam Optimizer in order to decrease the Cross Entropy 

Learning Rate = 0.0015
Batch Size = 128
Epochs = 20

These hyperparameters were chosen based on an iterative approach in order to find the best validation accuracy.

#### 4. Arriving at the final model and their results

In order to arrive at the final model, I first build the architecture intuitively in order to learn the most number of features as possible in the training. And based on this initial set-up, I tried to tune the hyperparameters mentioned in Step:3.
Once these were fixed, i tried to play around with the architecture parameters in order to fix one partular model. 
My final model results were:
* training set accuracy of 94.2%
* validation set accuracy of  95.5%
* test set accuracy of 99.2%

The LeNet architecture was retained throughout because i believed that this could be enoug to solve this image classification problem with a decent accuracy based on the lectures. The main thing adjusted were the height of each layer in order to learn many features as possible as this is crucial in identifiying traffic signs. Also a droput layer was introduced in order to avoid over-fitting the model.

### Test a Model on New Images

#### 1. 5 different Traffic signs were chosen from the web and were trained on the model

Here are five German traffic signs that I found on the web:


![alt text][image5] 

![alt text][image6] 

![alt text][image7]

![alt text][image8] 

![alt text][image9] 

![alt text][image10]

![alt text][image11] 

![alt text][image12] 

![alt text][image13]

![alt text][image14]


These images are generally difficult to classify because these are high resolution images and when these are shrunk to the desired size of 32x32, some information is usually lost. This causes difficulty in identifying distinct features from the image. Additionally, these parituclar labels were chosen as these labels had smaller number of training data compared to the other labels.

#### 2. Results on the new Test Images

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| Children Crossing   28  		|  	Children Crossing							|
| Dangerous Right   20  			| 		Speed Limit 120 KM/H						|
| Dangerous Left			19		|  	Dangerous Left				|
| General Caution	  18    		| 	General Caution		 				|
| Go Straight or Right	 36|   Go Straight or Right   							|
| Keep Left    	39	|  	Keep Left 								| 
| Road Work    	25		| 		Road Work								|
| Road Work    	25		| 		Road Work								|
| Slippery Road		23		|			Slippery Road								|
| Slippery Road			23	|			Wild Animals Crossing							|




The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Softmax probabilities of the Test Images

The Softmax probabilities for the test images are:

| Probabilities    | Classes |
|:---------------------:|:---------------------------------------------:| 
| [1.0000000e+00, 0.0000000e+00, 0.0000000e+00] | [28,  0,  1] |
| [1.0000000e+00, 1.0455428e-12, 2.4460186e-13] | [ 8,  1,  4] |
| [1.0000000e+00, 0.0000000e+00, 0.0000000e+00] | [20,  0,  1] |
| [1.0000000e+00, 0.0000000e+00, 0.0000000e+00] | [18,  0,  1] | 
| [9.9805415e-01, 1.9459319e-03, 0.0000000e+00] | [36, 38,  0] |
| [1.0000000e+00, 0.0000000e+00, 0.0000000e+00] | [39,  0,  1] |
| [1.0000000e+00, 1.3245671e-38, 0.0000000e+00] | [25, 22,  0] |
| [1.0000000e+00, 0.0000000e+00, 0.0000000e+00] | [25,  0,  1] |
| [1.0000000e+00, 0.0000000e+00, 0.0000000e+00] | [23,  0,  1] |
| [1.0000000e+00, 2.7554958e-19, 1.5261866e-21] | [31, 21, 25] |

Since the last image was misclassified as Wild Animals Crossing, it can be interpreted that the model has not learnt to predict Wild Animals Crossing as well as Road Work labels accurately. Meaning there is some ambiguity in classification as per the model. And this is more enhanced in images which undergo data loss due to shrinkage. So, we might neeed to feed additional data for these class for the model to better learn its features. As we can see in they hysteresis plot that these are also the labels with very less data points compared to the other labels. Similar statement can be said about predicting a Dangerous Right Curve as well. 


