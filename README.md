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
1. sudo apt install nvidia-cuda-toolkit
2. {{{ sudo apt install nvidia-cuda-toolkit }}}



REFERENCES:
- https://www.tensorflow.org/install/source#gpu
- https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d
- https://support.system76.com/articles/cuda/
- 
