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
#https://stackoverflow.com/questions/34875944/how-to-write-a-custom-loss-function-in-tensorflow
#https://cnvrg.io/keras-custom-loss-functions/

df = pd.read_csv("test.csv", sep=',') #, dtype=None) #, skiprows=[0])
#df.values
print (type(df))

#convert all nan values to 0
df = df.fillna(0)
print ('updated df ')
print (df.head())
print ('raw shape: ', df.shape)
#print (type(df))



#split train and testing data set by 80% 20%
(train_data, test_data) = train_test_split(df, test_size=0.2)
print ('train data shape:', train_data.shape)
print ('test data shape:', test_data.shape)
#convert the train and test arrayts to float 32
train_data = train_data.astype('float32').values
test_data = test_data.astype('float32').values

#look at the array shape of the first element in the array
print ('array shape of first element inside training array: ', train_data[0].shape)

#noremalize everything between 0 - and cast to float32
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
#print (min_val, max_val)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
#case the array again to float32 a
train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)


# The size of the latent space.
latent_dim = 4499

#building the encoder and decoder model as a tensor flow class
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

## Masked Mean Squared Error
def mmse(y_true, y_pred):
	mask = y_true != 0
	return K.mean(K.square(y_true[mask] - y_pred[mask]))


autoencoder = AutoEncoder(latent_dim)
#input_shape = (4499,1)
#autoencoder.build(input_shape)
autoencoder.compile(optimizer='adam', loss=mmse, metrics = ['mean_squared_error'])
            
#using a mse for now
#autoencoder.compile(optimizer='adam', loss='mse')
#autoencoder.summary()

fit = autoencoder.fit(train_data, train_data,
          epochs=40, 
          batch_size=100,
          verbose=1)

score = autoencoder.evaluate(test_data, test_data)
print('score is', score)

#plot the loss and accuracy for the chart save 
plt.plot(fit.history['mean_squared_error'], 'red')
plt.plot(fit.history['loss'])
plt.plot(fit.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['mean squared error', 'loss', 'accuracy'], loc='upper left')
plt.savefig("plotsaved.png")
plt.show()