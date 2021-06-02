import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as ku
import scipy.misc
from PIL import Image
from numpy import asarray

#https://www.tensorflow.org/guide/keras/save_and_serialize
#https://towardsdatascience.com/how-to-use-a-saved-model-in-tensorflow-2-x-1fd76d491e69
#https://www.tensorflow.org/tutorials/keras/classification

"""
#load the data targets
data = np.load("50x50flowers.images.npy")
targets = np.load("50x50flowers.targets.npy").astype("uint8")
#convert dataset to uint8 in order for it to be seen and used as an image 
#you need to divide dataset by 255 to be able to normalize the data
#otherwise the data sucks
image_data = data/255
#split the training and testing dataset with 80% used for training and 20% for tsting
(X_train, X_test, y_train, y_test) = train_test_split(image_data, targets, test_size=0.2)
#you need to shift the entire dataset by 1 since to categorical requires a starting point of 0
y_train = y_train - 1
y_test = y_test - 1
#turn it to catagorical from 0-17
y_train = ku.to_categorical(y_train, 17)
y_test = ku.to_categorical(y_test, 17)
"""

#https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays



def flower_model():
	THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
	print (THIS_FOLDER)
	h5_file = os.path.join(THIS_FOLDER, 'flowermodel.h5')
	print (h5_file)
	new_model = tf.keras.models.load_model(h5_file)
	# Check its architecture
	#print (new_model.summary())
	#ans = "model loaded"
	
	return new_model

def train_test_datax():
	#get the train test data
	THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
	print (THIS_FOLDER)
	img_path = os.path.join(THIS_FOLDER, 'image_1360.jpg')
	image = Image.open(img_path)
	#resize and save the image 
	new_image = image.resize((50,50))
	new_image.save('image_50.jpg')
	#convert image to array 
	data = asarray(new_image)
	print(type(data))
	# summarize shape
	print(data.shape)
	return data
#flower_model()

#loss, acc = new_model.evaluate(X_test, y_test, verbose=1)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

#print ('a single test subject to test')
# Grab an image from the test dataset.
#img = X_test[1]

#print(img.shape)
# Add the image to a batch where it's the only member.
#img = (np.expand_dims(img,0))
#print (img.shape)
#predictions_single = new_model.predict(img)

#print(predictions_single)
#b = np.argmax(predictions_single[0])
#print (b)