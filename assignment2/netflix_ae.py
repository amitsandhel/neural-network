#Amit Sandhel
#assignment 2 

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

#slide 25 of he gan class 
#replace False should probably be included 

#a variable to determine if we are going to use fit or train_on_batch
#default is tob ie train_on_batch
#the fit one is commented out
TRAINING="tob"
#TRAINING="fit" 

#open raw text file
#Note that the Header was manually removed from test.csv file
raw_df = pd.read_csv("test.csv", sep=',') #, dtype=None)

#convert all nan values to 0
raw_df = raw_df.fillna(0)


## Masked Mean Squared Error 
def mmse(y_true, y_pred):
	'''custom error function that removes zeros from the array
	'''
	mask = y_true != 0
	return K.mean(K.square(y_true[mask] - y_pred[mask]))

class AutoEncoder(Model):
	'''building the encoder and decoder model as a tensorflow class
	'''
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = tf.keras.Sequential([
			layers.Dense(8, activation="relu"),
			layers.Dropout(0.3),
			layers.Dense(16, activation="relu"),
			layers.Dropout(0.3),
			layers.Dense(32, activation="softmax"),
			layers.Dropout(0.3),
			layers.Dense(16, activation="relu"),
		])

		self.decoder = tf.keras.Sequential([
			layers.Dense(16, activation="relu"),
			layers.Dropout(0.3),
			layers.Dense(32, activation="relu"),
			layers.Dropout(0.3),
			layers.Dense(16, activation="relu"),
			layers.Dropout(0.3),
			layers.Dense(8, activation="softmax"),
			layers.Dropout(0.3),
			#if i dont have this then it crashes saying i have a size mismatch
			layers.Dense(4499, activation="relu"),
		])

	def call(self, input_features):
		encoded = self.encoder(input_features)
		decoded = self.decoder(encoded)
		return decoded


def setup_data(data_frame, var_training):
	'''this function will create the training test dataframe based on whether
	we are going to use a fit or a train_on_batch appraoch as we need an int
	or a float in some instances'''

	#split train and testing data set by 80% 20%
	(train_data, test_data) = train_test_split(raw_df, test_size=0.2)
	#convert the train and test data into arrays
	train_data = train_data.values 
	test_data = test_data.values 

	#find the min and max values of the array
	min_val = tf.reduce_min(train_data)
	max_val = tf.reduce_max(train_data)

	if var_training == "fit":
		print ('Running fit() ')
		#noremalize everything between 0
		train_data = (train_data - min_val) / (max_val - min_val)
		test_data = (test_data - min_val) / (max_val - min_val)
	else:
		#we have a batch the training data needs to be a int for now and not normalized
		#it will be normalized inside the epoch for loop 
		# we convert and normlaize the test data however
		print ('Runnig batch() ')
		#we need to convert the training and test data into int types to be able
		#to use the random.choice function
		train_data = train_data.astype(int)
		test_data = (test_data - min_val) / (max_val - min_val)

	return (train_data, test_data, max_val, min_val)

#run the setup_data() function and get the training, testing and max and min values of the data
train_data, test_data, max_val, min_val = setup_data(raw_df, var_training=TRAINING)

def fit_or_batch_trainer(train_data, test_data, var_training, max_max_val, min_val):
	'''function which can run the fit or train_on_batch model depending on user choice and display/calculate
	the score of the trained model
	'''
	autoencoder = AutoEncoder()
	autoencoder.compile(optimizer='RMSprop', loss=mmse, metrics = [mmse])

	if var_training == "fit":
		#we are running the fit model
		print ('Runnign fit training model')
		#this code works perfectly though can't tell how good the network actualy is
		#run for all epochs
		fit = autoencoder.fit(train_data, train_data,
		          epochs=70, batch_size=100, shuffle=True,
		          validation_data = (test_data, test_data),
		          verbose=1)
		#determine the score and accuracy of our model
		score = autoencoder.evaluate(test_data, test_data)
		print('score is', score)
	else:
		print ('Running train_on_batch')
		#loop through an epoch
		for epoch in range(1, 600):
			#extract random data from the training data 
			x = train_data[np.random.choice(train_data.shape[0], 100, replace=False)]
			#normalize the random extracted data
			x_data = (x - min_val) / (max_val - min_val)
			#run our autoencoder for training 
			loss = autoencoder.train_on_batch(x_data, x_data)
			print ('epoch: ', epoch, 'loss: ', loss)

		#determine the score and accuracy of our model 
		score = autoencoder.evaluate(test_data, test_data)
		print('score is', score)

#run the trainer model and determine the score as well
fit_or_batch_trainer(train_data, test_data, TRAINING, max_val, min_val)





"""
# Deprecated if you wish to graph it
#plot the loss and accuracy for the chart save 
plt.plot(fit.history['mean_squared_error'], 'red')
plt.plot(fit.history['loss'])
plt.title('model accuracy')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['mean squared error', 'loss'], loc='upper left')
plt.savefig("plotsaved2.png")
#plt.show()
"""

#REferecnes ignore this moved to bottom to make it easier
#https://keras.io/api/models/model_training_apis/
#https://www.tensorflow.org/tutorials/generative/autoencoder
#https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/
#https://blog.keras.io/building-autoencoders-in-keras.html
#https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
#https://www.tensorflow.org/tutorials/generative/autoencoder
#https://stackoverflow.com/questions/55908188/this-model-has-not-yet-been-built-error-on-model-summary
#https://stackoverflow.com/questions/44666098/how-to-implement-sparse-mean-squared-error-loss-in-keras
#https://github.com/keras-team/keras/issues/7065#issuecomment-318002081
#https://towardsdatascience.com/creating-custom-loss-functions-using-tensorflow-2-96c123d5ce6c
#https://stackoverflow.com/questions/34875944/how-to-write-a-custom-loss-function-in-tensorflow
#https://cnvrg.io/keras-custom-loss-functions/
#https://towardsdatascience.com/https-medium-com-chayankathuria-regression-why-mean-square-error-a8cad2a1c96f
#https://medium.datadriveninvestor.com/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop