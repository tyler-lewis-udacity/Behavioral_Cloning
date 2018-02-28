# **Behavioral Cloning Project**

### Writeup

---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[left]: ./images/left.jpg "Left Camera"
[center]: ./images/center.jpg "Center Camera"
[right]: ./images/right.jpg "Right Camera"
[correction_1]: ./images/correction_1.png "Correction 1"
[correction_2]: ./images/correction_2.png "Correction 2"
[correction_3]: ./images/correction_3.png "Correction 3"
[correction_4]: ./images/correction_4.png "Correction 4"
[mirror_left]: ./images/mirror_left.jpg "Mirror Left"
[mirror_right]: ./images/mirror_right.jpg "Mirror Right"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the following command in a bash terminal: 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network that is based on the NVIDIA model discussed in the course lecture material.  The model input layer is a 160x320x3 RGB image. The input images are then normalized from (0,255) to (-0.5,0.5) using the Keras Lambda function.  Only the middle portion of each image contains information relavent to the vehicle's position on the road, so a Cropping2D layer was used to remove the top portion of the images which makes training the model easier.  Also, the bottom portion of the image which showed only the hood of the car and provided no useful information was also excluded.  The model then uses five convolution layers with 2x2 striding.  A flatten layer and three fully connected layers complete the model.  The final output is a single neuron: the steering angle.  The model includes ELU (Exponential Linear Unit) layers to introduce nonlinearity, a Mean Squared Error loss function, and an Adam Optimizer.(model.py ine 80-97)      


#### 2. Attempts to reduce overfitting in the model

Five different types of data sets were used to ensure that the model was not overfitting. The model was tested by running it through the simulator at various desired speeds to ensure that the vehicle could stay on the track.  The longest simulator track test was 10 consecutive laps in which the car performed very well and only strayed outside the painted lane lines twice.  Also, the number of epochs was reduced from 10 to 3 in order to prevent overfitting.    

#### 3. Model parameter tuning

The model used an Adam Optimizer, so the learning rate was not tuned manually.  The best model was produced using 3 epochs based on the training and validation losses reported during training.    

#### 4. Appropriate training data

The model was trained and validated on 5 different data sets:  3 featured typical driving conditions and moaneuvers, 1 focussed on smooth cornering, and 1 focussed on corrective maneuvers.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to employ an already well known and reliable architecture.  At first, a modified LeNet architecture was used which produced terrible results.  The second model was based on the NVIDIA architecture developed by the NVIDIA self-driving car team.  It performed very well even on a small and simple training data set so I decided to use it for the model and focussed my efforts on imporoving the training data as much as possible.     

In order to gauge how well the model was working, I split my image and steering angle data into a training set(80%) and a validation set(20%).  I found that my first model with 10 epochs had a low mean squared error on the training set but a high mean squared error on the validation set with the validation loss increasing after the third epoch. This implied that the model was overfitting.  The mumber of epochs was reduced to 3 to reduce overfitting.  

The final step was to run the simulator to see how well the car was driving around track one. The first model, based on LeNet, resulted in the car driving around in circles.  After replacing the training model entirely, the vehicle drove much better but could not navigate turns at all.  After augmenting the training data with center, left, and right camera images and steering measurements (model.py lines 65-77), the vehicle could correctly maneuver through some of the turns.  After augmenting the images again by mirroring all images about their vertical center axis, the car could complete multiple laps without driving off the track.  By reducing the desired speed from 30 mph to 20 mph (drive.py line 47), ten full laps were completed without the car leaving the road.  

At the end of the process, the vehicle is able to drive autonomously around the track while remaining centered between the lane lines 99% of the time.  When the car does stray over the painted lane line, it corrects itself quickly.  Cornering is very smooth.

#### 2. Final Model Architecture

The final model consists of a convolution neural network that is based on the NVIDIA model discussed in the course lecture material.  The model input layer is a 160x320x3 RGB image. The input images are then normalized from (0,255) to (-0.5,0.5) using the Keras Lambda function.  Only the middle portion of each image contains information relavent to the vehicle's position on the road, so a Cropping2D layer was used to remove the top portion of the images which makes training the model easier.  Also, the bottom portion of the image which showed only the hood of the car and provided no useful information was also excluded.  The model then uses five convolution layers with 2x2 striding.  A flatten layer and three fully connected layers complete the model.  The final output is a single neuron: the steering angle.  The model includes ELU (Exponential Linear Unit) layers to introduce nonlinearity, a Mean Squared Error loss function, and an Adam Optimizer.      


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track while trying to remain centered between the lane lines.  The simulator features three cameras mounted to the vehicle.  An example of a left, center, and right camera image combonation is shown below:

![alt text][left]  ![alt text][center]  ![alt text][right] 


The fourth lap was completed while focusing on performing smooth and gradual cornering.  Shown below is a case where the car understeered while taking a sweeping left turn.  

The final lap focussed on correcting from over and under-steering positions that commonly occur while entering or leaving a turn.  Only the "correcting" portion of the driving was recorded.  (It would be a bad idea to train the model to ever drive towards the side of the road!)  These images show what the recovery looked like:

![alt text][correction_1]
![alt text][correction_2]
![alt text][correction_3]
![alt text][correction_4]


After the collection process, I had 24,108 training image data points. I then preprocessed this data by normalizing the images to have a mean of zero and equal variance.  I cropped the images to remove noise.  I edited the driving log to account for left, center, and right camera steering angle differences that were not recorded in the driving_log.csv.  To augment the data sat further, I also flipped images and angles about the center vertical axis. Here is an image that has then been flipped:

![alt text][mirror_left] ![alt text][mirror_right]


This was a quick and easy way to double the size of the training data.  Also, this ensured that the training data contained an even number of left and right turning examples.  (The track was circular which meant that the un-augmented contained more left turn examples than right turn examples.)

Shuffling of the data was done by Keras during training using the "shuffle=True" option.  

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the training and validation loss reports reported during traingin.  I used an Adam Optimizer so that manually training the learning rate wasn't necessary.
