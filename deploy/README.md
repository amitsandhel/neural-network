  GNU nano 2.9.3                                                                   READE.md                                                                             

HOW TO RUN the django example

This repo is a basic django application that runs an AI neural network tensorflow model
NOte that this looks ugly and basic because it is meant to be that way to simple be as basic as possible
to illustrate how to setup your neural network model inside a django applicatoin.

STEPS to Run
1. download the repo
2. cd into the deploy folder
3. If you need run and install the libraries from the requirements file via this command  pip install -r requirements.txt
3. run python manage.py runserver
4. This will now run the server locally
5. go to localhost url http://127.0.0.1:8000/myapp/simpleupload/
6. upload your picture and your done and click upload


NOTE: you can paste your own flower h5 file into the project and use that instead.
Most important file is views.py as that is where all the magic is happening.

The floewr images are located in this url you can download the zip file from there
https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html


I have attached and given some sample images from here for you to test and play with if need be
I noticed that sunflower seems to be the best results for my neural network but it is a very unique flower from the 17 flowers in the dataset

TODO:
run on aws at some point


