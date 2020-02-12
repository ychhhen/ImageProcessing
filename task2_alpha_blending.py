#Task 2: Alpha blending

import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util

import matplotlib.pyplot as plt

def hard_blending(im1, im2):
  """
  return an image that consist of the left-half of im1
  and right-half of im2
  """
  assert(im1.shape == im2.shape)
  h, w, c = im1.shape
  new_im = im1.copy()
  new_im[:,:(w//2),:] = im2[:,:(w//2),:]
  return new_im

def alpha_blending(im1, im2, window_size=0.2):
  """
  return a new image that smoothly combines im1 and im2
  im1: np.array image of the dimensions: height x width x channels; values: 0-1 
  im2: np.array same dim as im1
  window_size: what fraction of image width to use for the transition (0-1)
  """
  # useful functions: np.linspace and np.concatenate
  assert(im1.shape == im2.shape)
  # TODO: Put your code below
  height,width,channel = im1.shape

  #The default setting is horizontal stitching and blending
  def get_mask(img,start,stop):
    #define alpha mask array
    img_mask = np.zeros((height,width,len(start)),dtype=np.float)
    #define the window size:mid_width represents the width of blending area,
    #whereas left_width and right_width are widths of two original images.
    mid_width = np.int(window_size*width)
    left_width = np.int(((1-window_size)/2)*width)
    right_width = width-mid_width-left_width
    
    #create the alpha blending mask
    for i,(start,stop) in enumerate(zip(start,stop)):
        #set the value of alpha
        window = np.linspace(start,stop,mid_width)
        #set the value for non-blending area
        left_side = np.tile(start,left_width)
        right_side = np.tile(stop,right_width)
        #concatenate them
        temp_line = np.concatenate((left_side,window,right_side))
        img_mask[:,:,i] = np.tile(temp_line,(height,1))

    return img_mask
      

  #create the alpha mask for blending
  im_mask = get_mask(im1,(0,0,0),(255,255,255))
  im_mask = im_mask/255

  #use the equation for alpha blending
  blended_im = im1*im_mask+im2*(1-im_mask)

  return blended_im

if __name__ == "__main__":
  # TODO: Replace with your own images
  im1 = io.imread(path.join('images','day_London.jpg'))
  im1 = util.img_as_float(im1[:,:,:3])
  im2 = io.imread(path.join('images','night_London.jpg'))
  im2 = util.img_as_float(im2[:,:,:3])
  plt.figure(figsize=(15, 12)) 
  
  plt.subplot(221)
  plt.title('left image')
  plt.axis('off')
  plt.imshow(im1)

  plt.subplot(222)
  plt.title('right image')
  plt.axis('off')
  plt.imshow(im2)

  plt.subplot(223)
  plt.title('hard blending')
  plt.axis('off')
  plt.imshow(hard_blending(im1, im2))

  plt.subplot(224)
  plt.title('alpha blending')
  plt.axis('off') 
  alpha_1 = alpha_blending(im1, im2, window_size=0.2)
  plt.imshow(alpha_1)
  io.imsave('result/alpha_1.jpg', alpha_1)

  plt.show()
