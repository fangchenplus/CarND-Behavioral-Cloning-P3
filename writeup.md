# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./pictures/Architecture.png "Model Visualization"
[image2a]: ./pictures/left.jpg "Recovery Image"
[image2b]: ./pictures/center.jpg "Recovery Image"
[image2c]: ./pictures/right.jpg "Recovery Image"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode (no change from the provided file)
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results
* `video.mp4` a video recording of the vehicle driving autonomously at least one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In this project, I evaluated two CNN architectures, the LeNet model which I learnt from the number/traffic sign recognition, and a more sophisticated structure proposed by Nvidia in their paper [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The first model is simpler and severed as the basis to validate my programming pipeline. The second model is more tailored to this application (end to end learning) and was the final model I selected for this project. 

The overall strategy for deriving a model architecture was to use several convolution layers to recognize the pattern in the input pictures, and then some fully connected layers to generate the control output. Of course, RELU layers are put after every convolution layer to introduce nonlinearity. 

In order to gauge how well the model was working, I split my image and steering angle data into training and validation sets. I used 80% of the collected data as the training set and 20% as the validation set. I examined how the mean squared error on the training and validation set varied with different epochs. I found that the model settled quickly after just 3 epochs, after which the accuracy for the validation set started to drop. So I selected an epoch number of 3 in my final design.

The final step was to run the simulator to see how well the car was driving around track one. At first, there were a few spots where the vehicle fell off the track, for example, when the road turns and when there is no clear marking on one side of the road. To improve the driving behavior in these cases, I paid special attention to ensure enough training data in those cases were collected.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-94) consisted of a convolution neural network with the following layers and layer sizes. The CNN architecture is similar to Nvidia's end-to-end learning model cited below.

![alt text][image1]

```python
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

Using Keras, the building of CNN network is very simple. The format to use Conv2D function is `Conv2D(output size, kernel/window size, strides=(1, 1), activation=None)`. In my CNN, I have 5 convolution layers, and 4 dense layers. The depth of the convolution layers increases from 24 to 64, from the first to the fifth layer. The window size for the first three layers is 5x5, and for the last 2 convolution layers is 3x3. The stride also decreases from 2x2 to 1x1. RELU is always used as activation for the convolution layers.  The size of the fully connected dense layers gradually decreased from more than 100 to the final output size of 1 (steering).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, it is preferred to not only use center lane driving images but also left/right camera images. Though it is possible to record the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn, I found using  left/right camera images with corrected steering angle is sufficient for this purpose and is much easier to implement, not to mention the low feasibility to obtain recovery data in real road test. 

One group of the images from the left, center, and right cameras are provided below. A correction steering angle of 0.2 is used in my program, i.e., the virtual steering angles for the left and right images are generated by adding/subtracting 0.2 from the steering angle of the center image (real steering angle obtained from driving simulation).

| image from left camera | image from center camera | image from right camera |
| :--------------------: | :----------------------: | :---------------------: |
|  ![alt text][image2a]  |   ![alt text][image2b]   |  ![alt text][image2c]   |

To augment the data set, I also cropped and flipped images using Keras functions below.

```python
# data augmentation by flipping images and steering angles
augmented_images.append(cv2.flip(image,1))
augmented_measurements.append(measurement*-1.0) 

# data augmentation by normalization and cropping the useless top 70 pixel and bottom 25 pixcel of the image
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

Those have been proved to be very useful data augmentation techniques. Since the vehicle was driving counter-clockwise, the trained model had a tendency to turn left even when the road is straight. By flipping the image and steering angle, this bias was corrected.

The data is normalized in the model using a Keras lambda layer. The image cropping removes the useless top and bottom of the image, so the model can focus on the image part showing the road shape, not the vehicle hood or the sky and trees. This also reduces the image size, so the training is faster and memory need is smaller. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The optimization target is to minimize the mean square error (mse) between the model prediction and the simulated steering angle. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 after several trials and checking the accuracy trends of the training and validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

