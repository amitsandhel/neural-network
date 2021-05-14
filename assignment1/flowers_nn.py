import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras.models as km
import keras.layers as kl
import tensorflow.keras.utils as ku
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
#https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/#:~:text=The%20train%2Dtest%20split%20is,dividing%20it%20into%20two%20subsets.
#https://www.pluralsight.com/guides/data-visualization-deep-learning-model-using-matplotlib
#https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
#https://www.tensorflow.org/tutorials/images/classification
#https://www.tensorflow.org/tutorials/images/classification
#https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/


data = np.load("50x50flowers.images.npy")
targets = np.load("50x50flowers.targets.npy").astype("uint8")
print (data.shape)
print (targets.shape)
print (min(targets))
print (max(targets))
print (targets)

#convert dataset to uint8 in order for it to be seen and used as an image 
#you need to divide dataset by 255 to be able to normalize the data
#otherwise the data sucks
image_data = data/255

(X_train, X_test, y_train, y_test) = train_test_split(image_data, targets, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print ('k image data format: ', K.image_data_format())
#you need to shift the entire dataset by 1 since to categorical requires a starting point of 0
y_train = y_train - 1
y_test = y_test - 1
print ('categorical')
print (max(y_train), max(y_test))
print (min(y_train), min(y_test))
y_train = ku.to_categorical(y_train, 17)
y_test = ku.to_categorical(y_test, 17)

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# initialize an our data augmenter as an "empty" image data generator
aug = ImageDataGenerator()
aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest"
)


def make_model(numfm, numnodes, input_shape = (50, 50, 3), output_size = 17):
	# Initialize the model.
	model = km.Sequential()
	# Add a 2D convolution layer, with numfm feature maps.
	model.add(kl.Conv2D(numfm, kernel_size = (3, 3), input_shape = input_shape, activation = 'relu'))
	# Add a max pooling layer.
	model.add(kl.MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
	model.add(kl.Conv2D(numfm * 2, kernel_size = (3, 3), activation = 'relu'))
	# Add a max pooling layer.
	model.add(kl.MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
	# Convert the network from 2D to 1D.
	model.add(kl.Flatten())
	# Add a fully-connected layer.
	model.add(kl.Dense(numnodes, activation = 'tanh'))
	# Add the output layer.
	model.add(kl.Dense(output_size, activation = 'softmax'))
	# Return the model.
	return model

nn = make_model(20, 100)
nn.summary()

nn.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


#fit = nn.fit(X_train, y_train, epochs = 20, batch_size = 100, verbose = 1)
fit = nn.fit(
	x=aug.flow(X_train, y_train, batch_size = 100),
	#validation_data = (X_test, y_test),
	epochs = 20, verbose = 1
)

score = nn.evaluate(X_test, y_test)
print ('score ', score)

plt.plot(fit.history['accuracy'])
plt.plot(fit.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.savefig("plotsaved.png")
plt.show()







def make_plot():
	'''chart to show the first 10 images to see what we are looking at
	'''
	plt.figure(figsize=(10, 10))
	for i in range(9):
		#either divide by 255 to make it work or use uint8  
		dx = data[i]/255
		ax = plt.subplot(3, 3, i+1)
		#plt.imshow(dx)
		plt.imshow(dx) #dx.astype('uint8'))
		plt.title(str(targets[i]))
		plt.axis("off")
	plt.savefig("test3")
	plt.show()



'''
print (data[0])
print ('divide by 255')
print (data[0]/255)
print ('cast to unitint')
print (data[0].astype('uint8'))
'''
#make_plot()


