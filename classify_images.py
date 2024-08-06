from pyimg import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2


def preprocess_image(img):
    #swap the color channels from BGR to RGB, resize it, and scale 
    # the scale the pixel values to [0,1] range
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(image,(config.IMAGE_SIZE,config.IMAGE_SIZE),interpolation=cv2.INTER_AREA)
    img=img.astype("float32")/255.0

    # subtract ImageNet mean,divide by ImageNEt standard deviation,
    # set "channels first" ordering and add a batch dimension
    img-=config.MEAN
    img/=config.STD
    img=np.transpose(img,(2,0,1))
    img=np.expand_dims(img,0)

    return img    



ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the input image")
ap.add_argument("-m","--model",type=str,default="vgg16",
                choices=["vgg16","vgg19","inception","densenet","resnet"],
                help="name of the pretrained network to use")
args=vars(ap.parse_args())

MODELS={
    'vgg16':models.vgg16(pretrained=True),
    'vgg19':models.vgg19(pretrained=True),
    'inception':models.inception_v3(pretrained=True),
    'densenet':models.densenet121(pretrained=True),
    'resnet':models.resnet50(pretrained=True)
}

#load our network weights from disk ,flash it to the current
# device and set it to the evaluation mode
print(f"[INFO] loading {args['model']}")
model=MODELS[args["model"]].to(config.DEVICE)

model.eval()

print("[INFO] loading image....")
image=cv2.imread(args["image"])
orig=image.copy()
image=preprocess_image(image)

image=torch.from_numpy(image)
image=image.to(config.DEVICE)
print("[INFO] loading ImageNet labels...")

imagenetLabels=dict(enumerate(open(config.In_LABELS)))

# classify the image and extract the predictions
print(f"[INFO] classifying image with {args['model']}")
logits=model(image)
probabilities=torch.nn.Softmax(dim=1)(logits)
sortedProbability=torch.argsort(probabilities,dim=-1,descending=True)

#loop over the predictions and display the rank 5 predictions and
# corresponding probabilities to the terminal
for (i,idx) in enumerate(sortedProbability[0,:5]):
    print(f"{i}.{imagenetLabels[idx.item()].strip()}:{probabilities[0,idx.item()]*100:.2f}%")

    # draw the top prediction on the image and display the image to
    # our screen    
(label,prob)=(imagenetLabels[probabilities.argmax().item()],
probabilities.max().item())
cv2.putText(orig, f"Label: {label.strip()}, {prob * 100:.2f}%",
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)