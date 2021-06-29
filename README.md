# neural-network
Neural Network Assignment

How to setup the tensorflow for gpu

Hardware REquirements 
Nvidia GTX 1050Ti graphics card
16gb ram with 256gb SSD and a 1TB HD as well
lenovo legion y7000 laptop
PopOS linux Operating system full install (no windows or any dual boot pure linux)

ISSUE Detected: 
- We found that the most recent version of tensorflow 2.5.0 is not compatabile with our graphics card as tensorflow is only compatabile with Cuda 11 and we apparently have cuda 10.0. With the recent version not only was our graphcis card not detected but even the CPU run failed to work properly. 
- we are setting up tensorflow inside a virtualenv using pyenv (a virtualenv manager) the virtauev is called 

STEPS 
- these are the following installation and setup systems we did.

1. install cudat toolkit 
 ```
- sudo apt install nvidia-cuda-toolkit
```
2. After that we also installed the cuda toolkit using the the popos link direction (link: https://support.system76.com/articles/cuda/)
```
sudo apt install system76-cuda-latest
```
3. check if the nvidia driver actually got setup using the command nvcc --version 
```$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```
4. turns out CUDA is installed in a different path in 20.04, i.e. /usr/lib/cuda — which you can verify by running,
```
$ whereis cuda
cuda: /usr/lib/cuda /usr/include/cuda.h
```
5. install cuDNN from the nvidia link. Note we had to create our own login credentials for this. Turns out we installed cuda 10.2
6. then tar the file and unextract the files 
```
Download cuDNN v8.2.0 (April 23rd, 2021), for CUDA 10.2
cuDNN Library for Linux (x86)

$ tar -xvzf cudnn-10.1-linux-x64-v7.6.5.32.tgz
```
7. Next, copy the extracted files to the CUDA installation folder,
```
$ sudo cp cuda/include/cudnn.h /usr/lib/cuda/include/
$ sudo cp cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
```
9. Set the file permissions of cuDNN,
```
$ sudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
```
10. The CUDA environment variables are needed by TensorFlow for GPU support. To set them, we need to append them to ~/.bashrc file by running,
11. then afterwards reload the bashrc for new changes
```
$ echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc

$ source ~/.bashrc
```

From here onwards the cudad drivers should be setup and ready to go

# Tensorflow Setup
12. then i created my pyenv using python 3.8 and a virtualenv 
```
$ pyenv shell 3.8.9
$ pyenv activate tensorflow-nn
```
13. install tensorflow but we used an older verion 2.5 doesn't support cuda 10
```
pip install tensorflow==2.3.0
```
14. install alternative libraries
```
pip install matplotlib
pip install scikit-learn
```

# Pytorch Setup
- in my gaming laptop I need to put them on separate virtual environments 
15. now create a new virtualenv 
```
pyenv shell 3.9.4
pyenv activate pytorch-nn
```

16. pip install whatever additional libraries you need
17. the most important is pytorch lightning
```
pip install pytorch-lightning
```

- PyTorch check to see if the GPU is detected 
```
nami@erza:~/ExtraDrive/scinet/neural network$ pyenv activate pytorch-nn 
pyenv-virtualenv: prompt changing will be removed from future release. configure `export PYENV_VIRTUALENV_DISABLE_PROMPT=1' to simulate the behavior.
(pytorch-nn) nami@erza:~/ExtraDrive/scinet/neural network$ python
Python 3.9.4 (default, Apr  7 2021, 20:59:34) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.cuda.device_count())
1
>>> print(torch.cuda.current_device())
0
>>> print(torch.cuda.get_device_name(torch.cuda.current_device()))
GeForce GTX 1050 Ti
>>> print(torch.cuda.is_available())
True
>>> 

```

# REFERENCES:
- https://www.tensorflow.org/install/source#gpu
- https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d
- https://support.system76.com/articles/cuda/


# Screenshots
![Screenshot from 2021-05-15 21-08-57](https://user-images.githubusercontent.com/8258474/123721599-ab6f9080-d854-11eb-8ed4-ea13f9834300.png)
![Screenshot from 2021-05-15 21-09-10](https://user-images.githubusercontent.com/8258474/123721610-b0ccdb00-d854-11eb-8c53-3416483334e2.png)


