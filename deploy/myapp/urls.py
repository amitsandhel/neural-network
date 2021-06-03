""" Url configuration file """
#And then mysite/urls.py to look like this:

from django.urls import path, include
from . import views

urlpatterns = [
	path("simpleupload/", views.simple_upload, name="simpleUpload")
]