""" Url configuration file """
#And then mysite/urls.py to look like this:

from django.urls import path, include
from . import views

urlpatterns = [
	path("", views.index, name="index"),
	path("results/", views.results, name="results"),
	path("simpleupload/", views.simple_upload, name="simpleUpload")
]