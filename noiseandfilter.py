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
parnoise = "gauss" 

def noisy(noise_type,image,std=0.9,p=0.1,q=0.1):

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
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    if err == 0:
        return 20
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(err))


for i in range(len(x_train)):
      X_train[i] = noisy(parnoise, x_train[i])
      
      x[i] = psnr(X_train[i],x_train[i])
      avg = float(sum(x))/len(x)

print(avg)     
