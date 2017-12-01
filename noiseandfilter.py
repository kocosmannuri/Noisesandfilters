import noise
import numpy as np
import math



def noisy(noise_type,image,std=0.1,p=0.1,q=0.1):

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
