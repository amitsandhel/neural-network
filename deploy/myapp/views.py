from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import myapp.ans as mymodel
import numpy as np

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as ku
import scipy.misc
from PIL import Image
from numpy import asarray

#https://medium.com/saarthi-ai/deploying-a-machine-learning-model-using-django-part-1-6c7de05c8d7
#https://towardsdatascience.com/plant-ai-deploying-deep-learning-models-9dda5f6c1088?gi=b2f001d49049
#https://stackoverflow.com/questions/6086621/how-to-reduce-an-image-size-in-image-processing-scipy-numpy-python

# Create your views here.
def index(request):
	output = "hello bimbo erza"
	return HttpResponse(output)

def results(request):
	new_model = mymodel.flower_model()
	print (new_model)
	new_image = mymodel.train_test_datax()
	#print(new_image)
	
	# Add the image to a batch where it's the only member.
	img = (np.expand_dims(new_image,0))
	print (img.shape)
	#predict the 
	predictions_single = new_model.predict(img)

	print("the value is: ", predictions_single)
	b = np.argmax(predictions_single[0])
	print ("result is: " , b)

	return HttpResponse(b)


flower_name_list = ["ButterCup", "Colts Foot", "Daffodil", "Daisy","Dandelion",
					"Fritillary", "Iris", "Pansy", "Sunflower", "windflower",
					"snowdrop", "LilyValley", "Bluebell", "crocus", "Tigerlily",
					"Tulip", "Cowslip"] 

#https://simpleisbetterthancomplex.com/tutorial/2016/08/01/how-to-upload-files-with-django.html
def simple_upload(request):
	if request.method == 'POST' and request.FILES['myfile']:
		myfile = request.FILES['myfile']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)

		image = Image.open(myfile)
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
		new_model = mymodel.flower_model()
		#predict the 
		predictions_single = new_model.predict(img)

		print("the value is: ", predictions_single)
		b = np.argmax(predictions_single[0])
		print ("result is: " , b)
		name = flower_name_list[b-1]

		context = {'uploaded_file_url': uploaded_file_url,
					'ans': b, "name":name,
					'image_url': uploaded_file_url,
					}

		return render(request, 'myapp/simple_upload.html', context)

	return render(request, 'myapp/simple_upload.html')