# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/histogram.jpg "Histogram"
[image2]: ./figures/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./figures/sign1.jpg "Traffic Sign 1"
[image5]: ./figures/sign2.jpg "Traffic Sign 2"
[image6]: ./figures/sign3.jpg "Traffic Sign 3"
[image7]: ./figures/sign4.jpg "Traffic Sign 4"
[image8]: ./figures/sign5.jpg "Traffic Sign 5"
[image9]: ./figures/sign6.jpg "Traffic Sign 6"
[image10]: ./figures/sign7.jpg "Traffic Sign 7"

[image11]: ./figures/classification1.jpg "Traffic Sign Classification 1"
[image12]: ./figures/classification2.jpg "Traffic Sign Classification 2"
[image13]: ./figures/classification3.jpg "Traffic Sign Classification 3"
[image14]: ./figures/classification4.jpg "Traffic Sign Classification 4"
[image15]: ./figures/classification5.jpg "Traffic Sign Classification 5"
[image16]: ./figures/classification6.jpg "Traffic Sign Classification 6"
[image17]: ./figures/classification7.jpg "Traffic Sign Classification 7"

---


### Data Set Exploration

The project dataset consisted of pictures of traffic signs from the [German
Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

#### 1. Data Set Summary

A cursory examination using the pandas library reveals the following
statistics of the data set:

* The size of training set is 34,799 images
* The size of the validation set is 4,410 images
* The size of test set is 12,630 images
* Each image is a RBG image of shape 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization

The data is extremely imbalanced, with the number of images per class varying
from 180 to 2,010 as the following histogram shows:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing


I decided to begin by trying the LeNet-5 model on the traffic sign data, as
suggested by the project guidelines. Since LeNet-5 takes as input grayscale
images, I started off by writing a grayscaling function and converting all
images.

```
import numpy as np

def grayscale(rgb_img):
    weights = np.array([.2126, .7152, .0722])
    return np.average(rgb_img, axis=-1, weights=weights)
```

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I then normalized the images by "centering" them as in
```
X = (X - 128) / 128
```
for each of the train, validation and test sets.

I also considered data augmentation, which could be done by flipping some
images horizontally or vertically (or both). However, I suspected that one of the biggest challenges with the data is the fact that it is so imbalanced, and
image flipping might not necessarily correct it, but might actually
exacerbate it (for example, the least represented class, "Speed limit 20 km/h" would not be augmented as it has no line of symmetry, while one of the most numerous classes, "Speed limit 30 km/h" would benefit from this augmentation). I decided to leave out image augmentation in the first try.



#### 2. Model Architecture

My model was a beefed-up LeNet-5, which incorporated roughly double the
amount of nodes and also included regularization (dropout) as well as a
weighted cost function. It consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride,  valid padding, outputs 10x10x32  		
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x32                    |
| Fully connected		| Input 800, output 240					         |
| RELU 				    |         						                 |
| Droput                | keep_prob = 0.6                              |
| Fully connected		| Input 240, output 84					|
| RELU                  |                                   |
| Dropout               | keep_prob = 0.6   |
| Fully connected       | Input 84, output 43       |
| Softmax				|									|		|



#### 3. Model Training

I used the Adam optimizer (which runs gradient descent with momentum and
RMS prop), which has been known to work well on Computer Vision problems.

I settled on a dropout probability of 40% (60% keep probability), which seemed
to perform well for a range of similar network architectures.

I use a batch size of 128 and the standard learning rate of 0.001. I trained the model for 10 EPOCHS on an i7 CPU with 8 GB of memory.


#### 4. Solution Approach

I started out with the classical LeNet-5 network. This produced a validation
set accuracy of about 89%. I noticed that as the training iterated through epochs,
the training set accuracy was consistently about 10% higher than the validation
set accuracy. This is typically a symptom of overfitting, and in the case here
was likely due to the imbalance in the data. To address this, I decided to
incorporate dropout in the model.

Dropout immediately reduced the discrepancy between the training and testing
accuracy to a couple of percentage points. I experimented with different values of dropout probability and I noticed that aggressive values of keep_prob in the range of 50-70% performed well.

However, the training set accuracy deteriorated after adding dropout, so I decided to use a bigger model (same depth), by roughly doubling the number
of nodes in each convolution layer and also in the first fully connected one.

I have also tried to address the imbalance in the data by weighting the losses
in the cost function according to the label. I tried a weighting function
inversely proportional to the frequency of that class. This however did not have
a noticeable impact on performance (at least not on the network architecture
I tried), so my final model used uniform weights. I ended up with a validation set
accuracy of about 95%.


My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 95.4%
* test set accuracy of 93.6%



### Test a Model on New Images

#### 1. Acquiring New Images

I chose seven new images of German traffic signs from the web.


![][image4]
![][image5]
![][image6]
![][image7]
![][image8]
![][image9]
![][image10]

I expected the 'bumpy road' image to be more difficult to classify because it is
tilted, but otherwise I did not expect much difficulty.

#### 2. Performance on New Images

As I suspected, the 'bumpy road' image was misclassified (as 'Turn left ahead' - grayscaling in preprocessing did not help here..). To my surprise,
the 'Road work' sign was also misclassified as 'Bumpy Road'. There is a certain resemblance between these signs, both having two dark 'blobs' in the
center of the image, and the training set had only 330 samples on 'bumpy road', which perhaps explains the confusion between the two. Finally, the
"Speed limit 70 km/h" was misclassified as "Speed limit 30 km/h", which
probably means the model could benefit for more filters in the first
convolution layer (to allow it to pick more more small features).


Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| _Speed limit 70km/h_  | _Speed limit 30km/h_        			     	|
| Pedestrians     		| Pedestrians									|
| No entry				| No entry										|
| _Road Work_      		| _Bumpy Road_					 				|
| Right-of-way at intersection| Right-of-way at intersection            |
| _Bumpy Road_            | _End of Speed Limit 80km/h_                 |
| General Caution       | General Caution                               |


The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 57%. This compares un-favorably to the accuracy on the test set of 93%, however 7 data points are not enough to draw any statistical conclusions.

#### 3. Model Certainty - Softmax Probabilities

The trained model seemed virtually certain of its choice for the four images it
managed to classify correctly. The correct label featured as the second choice
in two of the misclassified cases (albeit a very far second, in one of them),
and it did not appear in the top 5 in the third misclassified image.

It is possible the the images I collected were somehow different from those in
the project's dataset (perhaps my conversion to 32x32 was different), or maybe the small number of new images (7) is not enough to draw statistical conclusions.

If I had more time to spend on this project, I would probably try a network which has one more convolution layer (so three in total) and more units in the early
layers (for small feature detection).



![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image15]

![alt text][image17]
