#assignment 2 


import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import mnist

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#https://www.tensorflow.org/tutorials/generative/autoencoder
#https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/
#https://blog.keras.io/building-autoencoders-in-keras.html
#https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
#https://www.tensorflow.org/tutorials/generative/autoencoder
#https://stackoverflow.com/questions/55908188/this-model-has-not-yet-been-built-error-on-model-summary
#https://stackoverflow.com/questions/44666098/how-to-implement-sparse-mean-squared-error-loss-in-keras
#https://github.com/keras-team/keras/issues/7065#issuecomment-318002081
#https://towardsdatascience.com/creating-custom-loss-functions-using-tensorflow-2-96c123d5ce6c

df = pd.read_csv("test.csv", sep=',') #, dtype=None) #, skiprows=[0])
#df.values
print (type(df))

#convert all nan values to 0
df = df.fillna(0)
print ('updated df ')
print (df.head())
print (df.shape)
print (type(df))
#convert and remove all nan to zero
#dataframe = np.nan_to_num(df)
#print (dataframe)



#split train and testing data set
(train_data, test_data) = train_test_split(df, test_size=0.2)
print (train_data.shape)
print (test_data.shape)
train_data = train_data.astype('float32').values
test_data = test_data.astype('float32').values
print (train_data.shape)
print (test_data.shape)
print ('value of 1')
print (train_data[0].shape)

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
print (min_val, max_val)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
print (train_data)
#print (train_data.tail())

# The size of the latent space.
latent_dim = 4499

mse = tf.keras.losses.MeanSquaredError()
loss = tf.keras.losses.mean_squared_error(train_data, train_data)
print (loss)

#building the encoder
class AutoEncoder(Model):
	def __init__(self, latent_dim):
		super(AutoEncoder, self).__init__()
		self.latent_dim = latent_dim
		
		self.encoder = tf.keras.Sequential([
			layers.Dense(3, activation="relu"),
			layers.Dense(16, activation="relu"),
			layers.Dense(8, activation="relu")
		])

		self.decoder = tf.keras.Sequential([
			layers.Dense(16, activation="relu"),
			layers.Dense(32, activation="relu"),
			layers.Dense(4499, activation="sigmoid"),
		])

	def call(self, input_features):
		encoded = self.encoder(input_features)
		decoded = self.decoder(encoded)
		return decoded

	def masked_mse(y_true,y_pred):
		#difference between true label and predicted label
		print ('xxx ', y_true, y_pred)

		error = y_true-y_pred    
		#square of the error
		sqr_error = K.square(error)
		#mean of the square of the error
		mean_sqr_error = K.mean(sqr_error)
		#square root of the mean of the square of the error
		sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
		print ('xxx' , sqrt_mean_sqr_error)
		#return the error
		return sqrt_mean_sqr_error



autoencoder = AutoEncoder(latent_dim)
#input_shape = (4499,1)
#autoencoder.build(input_shape)
autoencoder.compile(optimizer='adam', loss=masked_mse)
#autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#autoencoder.summary()

fit = autoencoder.fit(train_data, train_data,
          epochs=10, 
          batch_size=100,
          verbose=1)


score = autoencoder.evaluate(test_data, test_data)
print('score is', score)

