# P3 - Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
---
In this project we will build a convolution neural network to mimic human driving behavior in simulator provided by Udacity.  The goals/steps of this project are the following:

* Use the provided simulator to collect data
* Explore the data
* Data preprocessing and augmentation
* Build a generator
* Build a model
* Train and validate

[//]: # (Image References)
[image1]: ./images/collect_data.png
[image2]: ./images/dataframe.png
[image3]: ./images/explore_data_2.png
[image4]: ./images/cropping.png
[image5]: ./images/imbalance.png
[image6]: ./images/flipped.png
[image7]: ./images/steering.png
[image8]: ./images/random_contrast.png
[image9]: ./images/random_batch.png
[image10]: ./images/nvidia.png
[image11]: ./images/model.png
[image12]: ./images/
[image13]: ./images/

---
Track1
|[![Lake Track](track_1.mp4)](https://)

---

## Collect data

The driving simulator was provided by Udacity as part of the project.  There are total of 2 tracks that a simulator can be ran.  It is more important to collect the data that has good driving behavior so the model can be taught to generalize the driving on any course given.

![alt text][image1]

There simulator can be controlled with either mouse or keyboard.  Using keyboard changes the steering angle too dramatically when pressed which is not desirable for a good behavior driving.  Using a mouse or joystick gives the user much more control for smooth corners in simulator which would help our model to drive more smoothly.

## Explore the data

The data collected is divided into IMG folder which has all the frames of the driving, and driving_log.csv which has steering angle, throttle, brake, and speed of the vehicle for a given frame.  

![alt text][image2]

The images captured by the simulators are 160 x 320 pixels with RGB color channels.  Having three different camera angles for each frame will help immensely when we augment the data on later section for our training generators.  

![alt text][image3]

## Data Preprocessing and Augmentation

First step for data preprocessing was to crop out unnecessary parts of the image to reduce the computational strain during training.  The upper 50 pixels and lower 20 pixels were removed to remove unnecesary background and hood of the car.

![alt text][image4]

After the cropping, the images were resized using built in OpenCV function to fit our training model architecture which will be discussed later.

Next step to further improve the data set would be data augmentation.  In theory, the data can't be collected enough to cover all the situations, and if the model were trained on too much data, it would only raise the chance of overfitting without improving the model after certain threshold.  Below graph shows the imbalance of the dataset.

![alt text][image5]

The dataset has a left turn bias because the course that we trained on has dominantly left turns.  To offset this, we will flip images randomly, along with the corresponding steering angle to create a new data points that can be used to simulate right turns to train.

![alt text][image6]

The dataset also lacks the data points that recovers the vehicle coming in from the edge of the road to center of the road, because we tend to drive vehicle along the center line during collecting data.  We can augment existing data we have to simulate this behavior to further improve our model.  This is done by using left and right camera images.

![alt text][images7]

To simulate the steering angle of recovering back to the center, we will add 0.25 to the steering angle for left camera images and subtract 0.25 to the steering angle for right camera images.  The figure above is an example of the augmentation.

Last but not least, we will randomly augment images with different contrast to simulate different lighting conditions.

![alt text][image8]

## Build Generator

Since the data relatively memory intensive with 160 X 320 X 3 image with roughly the size of 50,000, we will use generators instead of preprocessing all the data at once and storing them to be more efficient at training our model.  We will generate batch size of 250 images to feed and train it to our model.  During this, to counter the bias toward the 0 steering angle which will hinder our model during corners, we will implement a condition to make each batch more balanced.   
```
            if abs(angle) <= 0.25:
                if np.random.uniform() > 0.9:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
            elif ((abs(angle) > 0.25) & (abs(angle) <= 0.5)):
                if np.random.uniform() > 0.9:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
            elif ((abs(angle) > 0.5) & (abs(angle) <= 0.75)):
                if np.random.uniform() > 0.2:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
            elif ((abs(angle) > 0.75) & (abs(angle) <= 1.0)):
                if np.random.uniform() > 0:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
```

This above code snippet will neutralize the heavy bias toward lower angle data in our batch generated datasets.  The below image shows more balaned dataset with much more higher angle points.

![alt text][image9]

## Build a model

Here, we will use Nvidia's neural network architecture.  

![alt text][image10]

As seen from the image, we will use normalization layer followed by 5 convolutional layers and 3 fully-connected layers to the output layer.  Unlike the Nvidia architecture, we will use a dropout layer between convolutional layers and fully-connected layers to reduce over-fitting.  'ELU' activation functions were used just like the Nvidia architecture.

Model used default adam optimizer, minimizing mean squared error between actual steering angle with predicted steering angle.  Roughly 50,000 images were in the dataset with generator generating 250 per batch with random augmentation to randomize.

![alt text][image11]

## Discussion

The future improvements to this project may include integrating P4 Advanced lane finiding to preprocessing image for this to dramatically reduce the hot pixels to reduce dimensionality and further stabilize the driving.

