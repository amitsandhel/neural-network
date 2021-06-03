import os
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import numpy as np
from numpy import asarray

import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as ku
import scipy.misc
from PIL import Image

#https://medium.com/saarthi-ai/deploying-a-machine-learning-model-using-django-part-1-6c7de05c8d7
#https://towardsdatascience.com/plant-ai-deploying-deep-learning-models-9dda5f6c1088?gi=b2f001d49049
#https://stackoverflow.com/questions/6086621/how-to-reduce-an-image-size-in-image-processing-scipy-numpy-python
#https://simpleisbetterthancomplex.com/tutorial/2016/08/01/how-to-upload-files-with-django.html
#https://www.tensorflow.org/guide/keras/save_and_serialize
#https://towardsdatascience.com/how-to-use-a-saved-model-in-tensorflow-2-x-1fd76d491e69
#https://www.tensorflow.org/tutorials/keras/classification
#https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays

# Create your views here.

def flower_model():
	"""load the tensorflow model we created in class"""
	# this is django stuff we need to get a path to the h5 files of interst
	THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
	print (THIS_FOLDER)
	h5_file = os.path.join(THIS_FOLDER, 'flowermodel.h5')
	print (h5_file)

	#load the tensorflow model and return it
	new_model = tf.keras.models.load_model(h5_file)
	return new_model



#this is the list of all 17 flower categories don't know if its the right one or not
#hence why the name output may be wrong
flower_name_list = ["ButterCup", "Colts Foot", "Daffodil", "Daisy","Dandelion",
					"Fritillary", "Iris", "Pansy", "Sunflower", "windflower",
					"snowdrop", "LilyValley", "Bluebell", "crocus", "Tigerlily",
					"Tulip", "Cowslip"] 


def modify_image(input_image):
	"""function to modify thie mage convert it to 50x50
	and then return it an array with a single elemnt in it
	"""
	image = Image.open(input_image)
	#resize and save the image 
	new_image = image.resize((50,50))
	new_image.save('image_50.jpg')
	#convert image to array 
	data = asarray(new_image)
	print(type(data))
	# summarize shape
	print(data.shape)
	img = (np.expand_dims(data,0))
	print (img.shape)
	return img

def simple_upload(request):
	"""function to upload a user flower image and determine the value and type 
	of flower
	"""
	#get the get response 
	if request.method == 'POST' and request.FILES['myfile']:
		#get the image object
		myfile = request.FILES['myfile']
		#save the file 
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)

		img = modify_image(myfile)

		#call and flower_model to get the tensorflow model
		new_model = flower_model()
		#predict the answer based on our input image
		predictions_single = new_model.predict(img)

		print("the value is: ", predictions_single)
		#get the highest value of the prediction
		result = np.argmax(predictions_single[0])
		print ("result is: " , result)
		name = flower_name_list[result-1]

		context = {'uploaded_file_url': uploaded_file_url,
					'ans': result, "name":name,
					'image_url': uploaded_file_url,
					}

		return render(request, 'myapp/simple_upload.html', context)

	return render(request, 'myapp/simple_upload.html')