# **Behavioral Cloning** 

## Writeup

[//]: # (Image References)

[image1]: ./plot_res.png "Visualization"

**Build a Behvaioral Cloning model to drive around a track**

The goals / steps of this project are the following:
* Collect training data by driving around the track
* Load the data set (see below for links to the project data set)
* Augment data by flipping the image and changing the sign of the steering angle
* Use left and right camera images and apply offset to the steering angles. The offset correction was tuned.
* Design, train and test a model architecture
* Use the model to drive the car around the track autonomously
* Summarize the results with a written report





## Rubric Points

### Writeup / README


The link to the project is [project code]
(https://github.com/rrsaikarthik3/Udacity_Self_Driving_Car_Project_4_Behavioral_Cloning/blob/main/model.py)




### Design and Test a Model Architecture

#### 1. Data Augmentation and Pre-Processing
1. Beofre pre-processing the data, the trianing data size was increased by flipping the existing images. And the corresponding steering angle's sign was changed.
2. The images loaded were converted to RGB format in order to comply with the autonomous driving image input format
3. Also, the left and right camera images were used for training by adding a correction factor to the center image's steering angle. This correction factor was tuned to have the optimal performance.
4. Theses images were then cropped to remove the top 50 rows and bottom 20 rows in order to reduce the actual training data to reduce the training time
5. The loaded images were then split into training and validation samples in order to not overfit the model
6. Once the images were cropped, each image was then normalized.






#### 2. The Training Architecture

The architecture used here is same as that of NVIDIA's 'End to End Learning for Self-Driving Cars' paper. The only modification made was that a dropout layer was added before the last convolution layer in order to avoid over-fitting.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x320x3 image   							| 
| Convolution 5x5     	| 1x1 stride,Same Padding,  outputs 38x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride,Same Padding,  outputs 17x77x36 	|   									|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride,Same Padding,  outputs 7x37x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride,Same Padding,  outputs 3x18x64 	|
| RELU					|												|
| Dropout  | Rate = 0.2 |
| Convolution 3x3     	| 1x1 stride,Same Padding,  outputs 1x8x64 	|
| RELU					|												|


| Fully connected		| 512 -> 100        									|
| Fully connected		| 100 -> 50        									|
| Fully connected		| 50 -> 10        									|
| Fully connected		| 10 -> 1        									|

 


#### 3. Training Model
To train the model, I used the Adam Optimizer in order to decrease the Mean Squared Error
Correction Factor to include Left and Right camera images = 0.5
Batch Size = 32
Epochs = 5

These hyperparameters were chosen based on an iterative approach in order to find the best validation accuracy.
Here is the result from the training indicating the losses after each epoch
![alt text][image1]

### Test a Model to run Autonomously on the Track

The model built could autonomously drive the car around the first track with ease keeping the car well within the road boundaries ensuring passenger safety
When the model was chosen to run on the second track, the model could not keep the car within the roads after a time. In order to work on this, one more batch of training data was fed by driving the car around the secioind track both clock-wise and anti-clockwise. This increased the training dataset by huge amount and thus increased the trianing time. After training with the new data, the model could keep itself within the roads of the secind track, but it did not have optimal performance to drive around the first track. So, that model is not included in the final submission here.

