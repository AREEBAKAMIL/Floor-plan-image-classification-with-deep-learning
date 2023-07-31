# Floor-plan-image-classification-with-deep-learning
Floor-plan-image-classification-with-deep-learning

In this project, I developed a binary classification model and a multi-class classification model using CNNS to identify and distinguish 'floorplans' from 'notfloorplans' images. 
 - 'Floorplans' include digitally drawn floorplans and hand-drawn floorplans
 - 'notfloorplan' include typed documents, handwritten documents and documents containing tabular data.

The dataset used is data I have collected from various UK-based property web pages. 
Link to dataset: https://drive.google.com/drive/folders/1lV-JuHDwk-xFDv5zehjhq4Y5C1gCOlaS?usp=sharing

For Binary classification, the data consists of 1500 images in total as follows:
- total training floorplans images: 500
- total training notfloorplans images: 500
  
- total validation floorplans images: 250
- total validation notfloorplans images: 250

For Multi-class classification, the data consists of 1500 images in total as follows:
- total training 'basic floor plan' images: 200
- total training hand drawn floor plan images: 200
- total training hand written document images: 200
- total training table document images: 200
- total training typed document images: 200


- total training  validation basic floor plan images: 100
- total training  validation hand drawn floor plan images: 100
- total training  validation hand written document images: 100
- total training  validation table document images: 100
- total training  validation typed document images: 100


# 1. Project background, problem statement, and solution overview

The Valuation Office Agency (VOA) gives the government the valuations and property advice needed to support taxation and benefits in England, Wales, and Scotland. The VOA's work includes (but is not limited to) compiling and maintaining lists of council tax (CT) bands for 26 million domestic properties.

As such, the VOA’s Documentum system contains terabytes of data which contain several pdf files. Some of these files contain floorplans. Manually identifying the presence of floor plans and extracting attributes is a very hectic, time-consuming, and resource-consuming task. Identifying whether a floor plan exists in a document is an essential step. The identified floor plans can then be used for analysis in which essential attributes are extracted for various business processes.

This project focused on identifying floorplans (including hand-drawn floor plans) and distinguishing them from other documents such as typed documents, handwritten documents and documents containing tabular data. To achieve this, I have used deep-learning techniques and developed CNN-based models as described below. 

# 2. Binary Image Classification

## Results: 
The training accuracy (in blue) gets close to 100% whereas the validation accuracy (in green) stalls close to 90%. Our validation loss reaches its minimum after only four epochs.

## Steps
1. Load the training and validation datasets
2. Building a Small Convnet
   - 3 X {convolution + relu + maxpooling} modules. The convolutions operate on 3x3 windows and the maxpooling layers operate on 2x2 windows. The first convolution extracts 16 filters, the following one extracts 32 filters, and the last one extracts 64 filters.
   - NOTE: This is a configuration that is widely used and known to work well for image classification. Also, since I have relatively few training examples (1,000), using just three convolutional modules keeps the model small, which lowers the risk of overfitting.
   - We will train our model with the binary_crossentropy loss because it's a binary classification problem and our final activation is a sigmoid.
   - Since this is a two-class classification problem, i.e. a binary classification problem, the network ends with a sigmoid activation, so that the output of the network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).
3. Resize the images - The images that go into convnet are 150x150 colour images
4. Normalize the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range).
5. Train on all 1000 images available, for 15 epochs, and validate on all 500 validation images.
6. Visualizing the intermediate representations to get a feel for what kind of features our convnet has learned. The raw pixels of the images become increasingly abstract and compact representations
7. Evaluating Accuracy and Loss for the model by plotting the training/validation accuracy and loss as collected during training


# 3. Binary Image Classification Optimized
This is an optimized version of the previous Binary Image classifier. I have used the following optimization techniques:
1. Data augmentation: A technique of artificially increasing the training set by creating modified copies of a dataset using existing data. 
2. Dropout regularization: It is a regularization technique for reducing overfitting. The term “dropout” refers to dropping out the nodes (input and hidden layer) in a neural network. Dropout simulates a sparse activation from a given layer, which interestingly, in turn, encourages the network to actually learn a sparse representation as a side-effect.
3. Train on all 1000 images available, for 30 epochs, and validate on all 500 validation images.
   
## Results:
The training accuracy (in blue) and validation accuracy (in orange) are both close to 95%.

## Steps
1. "Augment" the training data via a number of random transformations, so that at training time, the model will never see the exact same picture twice. This helps prevent overfitting and helps the model generalize better.
2. Add the data-augmentation transformations to the data preprocessing configuration.
3. Reconfigure our convnet architecture from the previous model to add some dropout, right before the final classification layer.
4. Evaluating Accuracy and Loss for the model by plotting the training/validation accuracy and loss as collected during training

# 4. Multi-class Image Classification
Since we need to distinguish floorplans from various different types of documents such as typed documents, handwritten documents and documents containing tabular data, a better approach would be to classify the images into their respective class as opposed classes as opposed to classifying them into a single category referred to as 'notfloorplans'  

## Results:
The training accuracy (in blue) and validation accuracy (in orange) are both close to 90%.


## Steps
1. Load the training and validation datasets
2. use data augmentation to augment the training data
3. Building a Small Convnet as developed in the previous model. Make sure to include the dropout layer
4. train the model for 50 epochs
5. Retrieve the list of accuracy results on training and validation data
