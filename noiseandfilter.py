import noise
import numpy as np
import noise
import numpy as np
import math
import cv2
import logging
import os
import sys
from scipy import misc
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

def noisy(noise_type,image,std=0.3,p=0.1,q=0.0):

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

def lowres(image):
    scaling_factors=(2, 3, 4)
    width, height, dimension = image.shape
    size = (height, width)
    for scaling_factor in scaling_factors:
            downscaled = misc.imresize(image, 1 / scaling_factor, 'bicubic', mode='RGB')
            rescaled = misc.imresize(downscaled, float(scaling_factor), 'bicubic', mode='RGB')
            low_res_image = np.clip(rescaled.astype(np.float32), 0.0, 255.0)
            low_res_image = cv2.resize(low_res_image, size)           
    return low_res_image


for i in range(len(x_train)):
      X_train[i] = noisy(parnoise, x_train[i])
      X_train[i] = lowres(X_train[i])
      x[i] = psnr(X_train[i],x_train[i])
      avg = float(sum(x))/len(x)

print(avg)      

