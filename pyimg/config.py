import torch

IMAGE_SIZE=224

#specify the ImageNet mean and standard deviation
'''
    Prior to passing an input image through our network 
    for classification, we first scale the image pixel 
    intensities by subtracting the mean and then dividing by 
    the standard deviation — this preprocessing is typical for 
    CNNs trained on large, diverse image datasets such as ImageNet.
'''
MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]

#determine the device we will be using for inference
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# specify path to the imageNet labels
In_LABELS= "ilsvrc2012_wordnet_lemmas.txt"


