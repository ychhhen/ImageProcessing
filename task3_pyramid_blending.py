#Task 3: Pyramid blending

import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt

from task2_alpha_blending import alpha_blending

def laplacian_pyramid(img, levels=4, sigma=1):
  """
  Decompose an image into a laplcaian pyramid without decimation (reducing resolution)
  img - greyscale image to decompose
  levels - how many levels of the pyramid should be created
  sigma - the standard deviation to use for the Gaussian low-pass filter
  return array of the pyramid levels, each the same resolution as input. The sum of these 
         images should produce the input image.
  """
  pyramid = []
  #TODO: Implement decomposition into a laplacian pyramid
  current_shape = img.shape
  #smoothed
  def smoothed(image, sigma):
      smoothed = filters.gaussian(image, sigma)
      return smoothed
  
  #define resize
  def resize(image, output_shape):
      input_shape = image.shape
      output_shape = tuple(output_shape)
      input_shape = input_shape+(1,)*(len(output_shape) - image.ndim)
      image = np.reshape(image, input_shape)

      return image
  
  #first layer
  smoothed_image = smoothed(img, sigma)
  pyramid.append(img-smoothed_image)
  #loop to construct layers
  for layer in range(levels-1):
      out_shape = tuple([current_shape])
      resized_image = resize(smoothed_image,out_shape)
      smoothed_image = smoothed(resized_image,sigma)
      current_shape = np.asarray(resized_image.shape)
      pyramid.append(resized_image-smoothed_image)
      if layer == levels-2:
          pyramid.append(smoothed_image)

  return pyramid

def pyramid_blending(im1, im2, levels=4, sigma=1):
  #TODO: Implement pyramid blending
  lp_im1 = laplacian_pyramid(im1,levels=4)
  lp_im2 = laplacian_pyramid(im2,levels=4)
  
  #test alpha blending by different masks
  a = np.array([0.5,0.4,0.3,0.3,0.2,0.2])
  b = np.array([0.2,0.2,0.3,0.3,0.4,0.4])#0.2,0.2,0.3,0.4,0.4
  c = np.array([0.1,0.2,0.4,0.2,0.2,0.3])
  i = 1
  blended_lp = []
  for im1,im2 in zip(lp_im1,lp_im2):
      blended_lp.append(alpha_blending(im1,im2,window_size=c[i]))
      i += 1 
      
  im1 = sum(blended_lp)
  return im1
  

if __name__ == "__main__":

  #Part 1: Laplacian pyramid decomposition
  #TODO: Replace with your own image 'cat_aligned.png' 'day_London.jpg'
  im = io.imread(path.join('images','house_r.jpg'))
  im = util.img_as_float(im[:,:,:3])
  im = color.rgb2grey(im)
  pyramid = laplacian_pyramid(im, levels=4)

  plt.figure(figsize=(3*len(pyramid), 3))
  grid = len(pyramid) * 10 + 121
  for i, layer in enumerate(pyramid):
    plt.subplot(grid+i)
    plt.title('level {}'.format(i))
    plt.axis('off')
    if i == (len(pyramid)-1):
      io.imshow(layer)
    else:
      plt.imshow(layer)
    
  plt.subplot(grid+len(pyramid))
  plt.title('reconstruction')
  plt.axis('off')
  io.imshow(sum(pyramid))

  plt.subplot(grid+len(pyramid)+1)
  plt.title('differences')
  plt.axis('off')
  plt.imshow(im - sum(pyramid))
  plt.show()  

  # Part 2: Pyramid blending
  #TODO: Replace with your own images
  im1 = io.imread(path.join('images','house_r.jpg'))
  im1 = util.img_as_float(im1[:,:,:3])
  im2 = io.imread(path.join('images','house_l.jpg'))
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
  plt.title('alpha blend')
  plt.axis('off')
  plt.imshow(alpha_blending(im1, im2, window_size=0.3))
  io.imsave('results/ab_1.jpg', p_b)

  plt.subplot(224)
  plt.title('pyramid blend')
  plt.axis('off')
  p_b = pyramid_blending(im1, im2)
  plt.imshow(p_b)
  io.imsave('results/pb_1.jpg', p_b)
  plt.savefig('results/pb_1.jpg')
  plt.show()
