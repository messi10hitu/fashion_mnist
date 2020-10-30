import tensorflow as tf
from tensorflow import keras  # keras is an API for tensorflow which allow us to write less code
import numpy as np
import matplotlib.pyplot as plt

"""
1.Import the Dataset
2.Preprocess the Dataset
3.Build the Model
    a) input layer
    b) hidden layer
    c) output layer
4.Compile the Model
    a) Optimizer
    b) Loss
    c) Metrics
5.Train the Model
    a) Feed the Model
    b) Evaluate the Model
6.Predict the Model 
Keras is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models.
It wraps the efficient numerical computation libraries Theano and TensorFlow 
and allows you to define and train neural network models in just a few lines of code
"""

# IMPORT THE DATASET
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
# print(train_images[1])
print(len(train_images))
print(len(test_images))
print(len(train_labels))
print(len(test_labels))
# The train_images and train_labels arrays are the training set—the data the model uses to learn.
# The model is tested against the test set, (the test_images, and test_labels) arrays.
# each images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255.
# The labels are an array of integers, ranging from (0 to 9)correspond to the class of clothing the image represents:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# PREPROCESS THE DATA
# We do this to get the values between 0 and 1 to minimize the calculations
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# plt.figure(figsize=(5, 5))
# plt.imshow(train_images[0], cmap=plt.cm.binary)  # here cmap helps us to get the image in grey scale
# plt.show()


# BUILD THE MODEL

# Setup the layers:
# 1.input
# 2.hidden
# 3.output

"""Here we're specifying the first convolution. We're asking keras to generate 64 filters for us. These filters are 3 
by 3, their activation is relu, which means the negative values will be thrown way, and finally the input shape is as 
before, the 28 by 28. That extra 1 just means that we are tallying using a single byte for color depth. As we saw 
before our image is our gray scale, so we just use one byte. """
"""
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),  # transforms the format of the images from a two-dimensional array
    # (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)
    keras.layers.Dense(128, activation="relu"),  # These are densely connected, or fully connected, neural layers.
    # The first Dense layer has 128 nodes (or neurons). This is the Hidden layer "relu" = rectified linear unit.
    # This hidden layer is used to manage our bias and the weights and figure out the patter for accurate results.
    keras.layers.Dense(10, activation="softmax")])  # The second (and last) layer returns a logits array with length
# of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
# This is the output layer which consist of 10 labels (0,9) and we wnt our output in the reangeB?w (0,1)

# SOFTMAX = The Softmax regression is a form of logistic regression that normalizes an input value
# into a vector of values that follows a probability distribution whose total sums up to 1.
# The output values are between the range [0,1].The function is usually used to compute losses
# that can be expected when training a data set


# compile the model:
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Optimizers ==> update the weight parameters to minimize the loss function
# This is how the model is updated based on the data it sees and its loss function.

# Loss function ==> measures how accurate the model is during training.
# You want to minimize this function to "steer" the model in the right direction.

# Metrics ==> Used to monitor the training and testing steps.

# TRAIN THE MODEL:

# feed the model:
model.fit(train_images, train_labels, epochs=15)
# epochs tells us how many times the model is going to see the (train_images, train_labels) to get the better accuracy

# evaluate the accuracy:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("loss: ", test_loss)
print("Acurracy: ", test_acc)

# PREDICTION:

prediction = model.predict(test_images)
print(prediction)  # it gives us the prediction of 10000 test images
print(len(prediction))
print("-----------------------")
print(prediction[0])  # it gives us the prediction of 1st image with value of 10 labels inside it.
print(np.argmax(prediction[0]))  # it gives the maximum value on that prediction with in 10 labels
# print(np.argmax(prediction[1]))
# print(np.argmax(prediction[2]))
# print(np.argmax(prediction[3]))
# print(np.argmax(prediction[4]))
# print(np.argmax(prediction[5]))
# print(np.argmax(prediction[6]))
# print(np.argmax(prediction[1]))
print(class_names[np.argmax(prediction[0])])
# print("----------------")
# SAVE THE MODEL:
model.save("fashion_model.h5")
"""

# load the model
model = keras.models.load_model('fashion_model.h5')
# model.summary()

# PREDICTION:
prediction = model.predict(test_images)
# print(prediction)  # it gives us the prediction of 10000 test images
# print(len(prediction))

# for i in range(30):
#     print("actual: ", class_names[test_labels[i]])
#     print("prediction:", class_names[np.argmax(prediction[i])])
#
# for i in range(5):
#     plt.grid(False)
#     plt.imshow((np.squeeze(test_images[i])), cmap=plt.cm.binary)
#     plt.xlabel("Actual: " + class_names[test_labels[i]])
#     plt.title("Prediction:" + class_names[np.argmax(prediction[i])])
#     plt.show()
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow((np.squeeze(test_images[i])), cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
#
#

