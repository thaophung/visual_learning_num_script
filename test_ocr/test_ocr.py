import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pytesseract
from scipy.misc import toimage,imshow
import cv2



def get_img(n, img_sz):
  im = np.zeros(img_sz)
  pil_im = Image.fromarray(im)
  draw = ImageDraw.Draw(pil_im)
  font = ImageFont.truetype("DejaVuSans.ttf", 12)
  sz = font.getsize(str(n))
  draw.text((img_sz[1]-sz[0], 2), str(n), font=font)
  im = np.asarray(pil_im)
  return im

def get_action(n1, n2):
  return n1 + n2



NMax = 9999999
img_sz = (15,60)
ocr_string = open("ocr_string.txt","w")
data = np.zeros((img_sz[0], img_sz[1],1))
#mu = np.load('mu.npy')
count = 0.
for i in range(1000):
    n1 = np.random.randint(1000000,NMax)
    data[:,:,0] = get_img(n1, img_sz)
    img = toimage(data[:,:,0])
    img.save(str(i) + ".jpg")
    
    img = cv2.imread(str(i) + ".jpg")
    img_resize = cv2.resize(img, None, fx=5, fy=5)
    img_resize = Image.fromarray(np.uint8(img_resize))
    ocr_pred = pytesseract.image_to_string(img_resize, config='digits')
    img_resize.save(ocr_pred+'.jpg')

    if str(n1) == ocr_pred:
        count += 1
        #print count
    else:
        print i
        print ocr_pred
        print str(n1)
accuracy = count/1000
print accuracy


