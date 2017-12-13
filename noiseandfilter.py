import noise
import numpy as np
import math
import cv2
import logging
import os
import sys
from mytools.gtsrb import GTSRB

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
GTSRB_PATH = os.path.join(os.path.dirname(os.path.abspath('C:/Users/osman/Desktop/classifier/GTSRB')), 'GTSRB')
logging.debug("Loading GTSRB data...")
img_set = GTSRB(root_dir=GTSRB_PATH)
(x_train, y_train), (x_test, y_test) = img_set.get_pix_data()

x = []
x = list(range(len(x_train)))
X_train = []
X_train = list(range(len(x_train)))

parnoise = "quantization" 

def noisy(noise_type,image,std=0.9,p=0.9,q=0.4):

    if noise_type == "gauss":
        a = noise.GaussianNoise(std,scale = [0,255])
        noisy = a.apply(image)
        return noisy
    elif noise_type == "s&p":
        b = noise.SaltAndPepperNoise(p,scale=[0,255])
        noisy = b.apply(image) 
        return noisy
    elif noise_type == "quantization":
        c = noise.QuantizationNoise(q,scale =[0,255])
        noisy = c.apply(image)
        return noisy

def psnr(imageA, imageB):
    errR = np.sum((imageA[:,:,0].astype("float") - imageB[:,:,0].astype("float")) ** 2)
    errR /= float(imageA.shape[0] * imageA.shape[1])
    errG = np.sum((imageA[:,:,1].astype("float") - imageB[:,:,1].astype("float")) ** 2)
    errG /= float(imageA.shape[0] * imageA.shape[1])
    errB = np.sum((imageA[:,:,2].astype("float") - imageB[:,:,2].astype("float")) ** 2)
    errB /= float(imageA.shape[0] * imageA.shape[1])
    err = (errR + errG +errB)/3
    if err == 0:
        return 20
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(err))


for i in range(len(x_train)):
      X_train[i] = noisy(parnoise, x_train[i])
      
      x[i] = psnr(X_train[i],x_train[i])
      avg = float(sum(x))/len(x)

print(avg)  
