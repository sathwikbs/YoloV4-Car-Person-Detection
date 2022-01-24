# YoloV4-Car-Person-Detection

## Links to dataset and framework

YOLO v4 : https://github.com/AlexeyAB/darknet
Paper YOLO v4: https://arxiv.org/abs/2004.10934
Manual: https://github.com/AlexeyAB/darknet/wiki
pylabel : https://pypi.org/project/pylabel/
pre-trained weights : https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

## About the model

YOLO stands for You Only Look Once, which is an object detection model used in deep learning use cases.
There are mainly two main groups of object detection models: Two-Stage Detectors and One-Stage Detectors. 
YOLO is one of the One-Stage Detectors (You only look once, one-stage detection).
Given below is the architecture of the model:

## Primary Anlaysis

There are 2239 images in the dataset with annotations for both car and person in COCO json format. To convert the json to Yolo annotation format, we used the pylabel library.
The data was separated into 90% train and 10% test.
We configured the following files within the Yolo library for custom object detection:

### yolov4.cfg
I recommend having batch = 64 and subdivisions = 16 for ultimate results.Since I ran into memory issue I set batch_size to 32 and subdivision to 8.

Make the rest of the changes to the cfg based on how many classes you are training your detector on.

Note: I set my max_batches = 6000, steps = 4800, 5400, I changed the classes = 2 in the three YOLO layers and filters =  in the three convolutional layers before the YOLO layers.

How to Configure Your Variables:

width = 416

height = 416 (these can be any multiple of 32, 416 is standard, you can sometimes improve results by making value larger like 608 but will slow down training)

max_batches = (# of classes) * 2000 (but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000, however detector for 5 classes would have max_batches=10000)

steps = (80% of max_batches), (90% of max_batches) (so if your max_batches = 10000, then steps = 8000, 9000)

filters = (# of classes + 5) * 3 (so if you are training for one class then your filters = 18, but if you are training for 4 classes then your filters = 27)

### obj.names and obj.data

obj.names is a file where you will have one class name per line

obj.data contains number of classses,training data path ,test data path and backup path ,where we will save the weights to of our model throughout training.

### train.txt and test.txt

train.txt and test.txt files holds the relative paths to all our training images and test images.

