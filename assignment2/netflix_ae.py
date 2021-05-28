#assignment 2 


import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import mnist

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import numpy.random as npr
#slide 25 of he gan class 
#replace False should probably be included 


df = pd.read_csv("test.csv", sep=',') #, dtype=None) #, skiprows=[0])
#df.values
print (type(df))

#convert all nan values to 0
df = df.fillna(0)
print ('updated df ')
print (df.head())
print ('raw shape: ', df.shape)

#split train and testing data set by 80% 20%
(train_data, test_data) = train_test_split(df, test_size=0.2)
print ('train data shape:', train_data.shape)
print ('test data shape:', test_data.shape)
#convert the train and test arrayts to float 32
train_data = train_data.values #.astype('float32').values
test_data = test_data.values #.astype('float32').values

#look at the array shape of the first element in the array
print ('array shape of first element inside training array: ', train_data[0].shape)

#noremalize everything between 0
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
print ('min max value ', min_val, max_val)
#train_data = (train_data - min_val) / (max_val - min_val)
#test_data = (test_data - min_val) / (max_val - min_val)
print ('normalized  data ')
print (train_data)




## Masked Mean Squared Error 
def mmse(y_true, y_pred):
	#custom error function that removes zeros
	mask = y_true != 0
	return K.mean(K.square(y_true[mask] - y_pred[mask]))

#building the encoder and decoder model as a tensor flow class
class AutoEncoder(Model):
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = tf.keras.Sequential([
			layers.Dense(8, activation="relu"),
			layers.Dense(16, activation="relu"),
			layers.Dense(32, activation="softmax"),
			layers.Dense(16, activation="relu"),
		])

		self.decoder = tf.keras.Sequential([
			layers.Dense(16, activation="relu"),
			layers.Dense(32, activation="relu"),
			layers.Dense(16, activation="relu"),
			layers.Dense(8, activation="softmax"),
			#if i dont have this then it crashes saying i have a size mismatch
			layers.Dense(4499, activation="relu"),
		])

	def call(self, input_features):
		print ('call feature ', input_features)
		encoded = self.encoder(input_features)
		decoded = self.decoder(encoded)
		return decoded


"""
#this code works perfectly though can't tell how good the network actualy is
autoencoder = AutoEncoder()
autoencoder.compile(optimizer='RMSprop', loss=mmse, metrics = ['mean_squared_error'])
#run for all epochs
fit = autoencoder.fit(train_data, train_data,
          epochs=70, batch_size=100, shuffle=True,
          validation_data = (test_data, test_data),
          verbose=1)
"""

#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop

#train on batch attempt
autoencoder = AutoEncoder()
autoencoder.compile(optimizer='RMSprop', loss=mmse, metrics = ['mean_squared_error'])
#to use the train_on_batch
#onvert the train_data into an integer

train_data = train_data.astype(int)
test_data  = test_data/5 #.astype(int)
for epoch in range(1, 50):
	x = train_data[np.random.choice(train_data.shape[0], 100, replace=False)]
	#y= test_data[np.random.choice(test_data.shape[0], 100, replace=False)]
	#print (x.shape)
	x_data = x/5
	#y_data = y/5
	loss = autoencoder.train_on_batch(x_data,x_data)
	print ('epoch: ', epoch, 'loss: ', loss)


score = autoencoder.evaluate(test_data, test_data)
print('score is', score)


"""
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
