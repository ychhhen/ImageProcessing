# Task 1: Image enhancement
import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util
from skimage import img_as_ubyte
from skimage import img_as_float

import matplotlib.pyplot as plt

# useful functions: np.bincount and np.cumsum

def equalise_hist(image, bin_count=256):
  """
  Perform histogram equalization on an image and return as a new image.

  Arguments:
  image -- a numpy array of shape height x width, dtype float, range between 0 and 1
  bin_size -- how many bins to use
  """
  # TODO: your histogram equalization code
  #define arrays
  image = img_as_ubyte(image)
  row,col = image.shape
  new_image = np.zeros((row,col),dtype='uint8') 

  # compute the value of each grayscale,and save in image_hist 
  image_hist = np.bincount(image.flatten(), minlength=(bin_count))

  # normalise n[]
  norm_arr = (np.cumsum(image_hist)/(image.size))*(bin_count-1)
  norm_arr = norm_arr.astype('uint8')
  
  #Compute a normalized cumulative histogram
  for x in range(row):
      for y in range(col):
          new_image[x,y] = norm_arr[image[x,y]]
          
  return new_image

def he_per_channel(img):
  # Perform histogram equalization separately on each colour channel. 
  # TODO: put your code below
  #define
  img = img_as_ubyte(img)
  bin_count = 256
  row,col,channel = img.shape
  new_img = np.zeros((row,col,channel),dtype='uint8') 

  #use the same method as function: equalise_hist(image, bin_count=256)
  for i in range(3):
      # compute the value of each channel,and save in image_hist 
      image_hist = np.bincount(img[:,:,i].flatten(), minlength=(bin_count))
    
      # normalise n[]
      norm_arr = (np.cumsum(image_hist)/((img[:,:,i].size))*(bin_count-1))
      norm_arr = norm_arr.astype('uint8')
      
      #Compute a normalized cumulative histogram
      for x in range(row):
          for y in range(col):
              new_img[x,y,i] = norm_arr[img[x,y,i]]
          
  return new_img
  

def he_colour_ratio(img):
  # Perform histogram equalization on a gray-scale image and transfer colour using colour ratios.
  # TODO: put your code below
  bin_count = 256
  gray_img = img_as_ubyte(color.rgb2gray(img))
  row,col,channel = img.shape
  #new_gray_image = np.zeros((row,col),dtype='uint8')
  new_img = np.zeros((row,col,channel)) 

  # compute the value of each grayscale,and save in image_hist 
  image_hist = np.bincount(gray_img.flatten(), minlength=(bin_count))

  # normalise n[]
  norm_arr = (np.cumsum(image_hist)/(gray_img.size))*(bin_count-1)
  norm_arr = norm_arr.astype('uint8')
  
  #restore color from color ratio: Rnew(x; y) = Vnew(x; y)*Rold(x; y)/Vold(x; y)
  for i in range(3):
        for x in range(row):
            for y in range(col):
                #Compute a normalized cumulative histogram：new_gray_image[x,y] = norm_arr[gray_img[x,y]]
                new_img[x,y,i] = ((norm_arr[gray_img[x,y]])*img[x,y,i])/gray_img[x,y]
  
  new_img = img_as_float(new_img)
  
  return new_img
  

def he_hsv(img):
  # Perform histogram equalization by processing channel V in the HSV colourspace.
  # TODO: put your code below
  bin_count = 256
  hsv_img = img_as_ubyte(color.rgb2hsv(img))
  row,col,channel = hsv_img.shape
  #new_hsv_image = np.zeros((row,col,channel),dtype='uint8')
  new_img = np.zeros((row,col,channel)) 

  # compute the value of each grayscale,and save in image_hist 
  val_img_hist = np.bincount(hsv_img[:,:,2].flatten(), minlength=(bin_count))

  # normalise n[]
  norm_arr = (np.cumsum(val_img_hist)/(hsv_img[:,:,2].size))*(bin_count-1)
  norm_arr = norm_arr.astype('uint8')
  
  for x in range(row):
      for y in range(col):
          hsv_img[x,y,2] = norm_arr[hsv_img[x,y,2]]
  
  new_img = color.hsv2rgb(hsv_img)
  
  return new_img

if __name__ == "__main__":

    # TODO: Change the file to your own image
    img_name = 'fog rainbow.jpg'
    
    test_im = io.imread(path.join('images',img_name))
    
    test_im_gray = color.rgb2gray(test_im)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.title('Original image')
    plt.axis('off')
    plt.imshow(test_im_gray,cmap='gray')
    #io.imsave('result/g_0.jpg', test_im_gray)
    
    plt.subplot(122)
    plt.title('Histogram equalised image')
    plt.axis('off')
    plt.imshow(equalise_hist(test_im_gray),cmap='gray')
    io.imsave('result/g_1.jpg', equalise_hist(test_im_gray))
    
    plt.show()
    
    
    
    test_im = io.imread(path.join('images',img_name))
    test_im = util.img_as_float(test_im)
    plt.figure(figsize=(15, 12))
    
    plt.subplot(321)
    plt.title('Original image')
    plt.axis('off')
    io.imshow(test_im)
    
    plt.subplot(322)
    plt.title('Each channel processed seperately')
    plt.axis('off')
    io.imshow(he_per_channel(test_im))
    io.imsave('result/rgb_1.jpg', he_per_channel(test_im))
    
    
    plt.subplot(323)
    plt.title('Gray-scale + colour ratio')
    plt.axis('off')
    io.imshow(he_colour_ratio(test_im))
    io.imsave('result/rgb_2.jpg', he_colour_ratio(test_im))
    
    plt.subplot(324)
    plt.title('Processed V in HSV')
    plt.axis('off')
    io.imshow(he_hsv(test_im))
    io.imsave('result/rgb_3.jpg', he_hsv(test_im))
    
    plt.show()
