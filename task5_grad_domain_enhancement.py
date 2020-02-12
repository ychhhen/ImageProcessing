#Task 5: Gradient domain image enhancement
import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy import signal
import skimage
from scipy.interpolate import interp1d
from skimage import color

from task4_grad_domain import img2grad_field, reconstruct_grad_field

def grad_domain_enhancement(im_gray, a):
    """Implement task4(reconstructing image from gradient field) to enhance images
    im_gray - The grayscale image for enhancement
    a - Alpha value for f(G) = 1/(G^alpha+ε)
    """
    #Hint: Use reconstruct_grad_field from the previous task
    #Original gradient field
    G = img2grad_field(im_gray)
    G_x = G[:,:,0]
    G_y = G[:,:,1]
    #generate the gradient magnitude for the original image
    Gm_ = np.sqrt(np.sum(G*G, axis=2))
        ##create the linear function for computing enhanced gradient magnitude
    x = [-1.0,-0.1,0,0.1,1.0]
    y = [-1.0,-0.3,0,0.3,1.0]
    f = interp1d(x,y, fill_'extrapolate')
    #generate the enhanced gradient magnitude G'
    Gm = f(Gm_)
    for i in range(G.shape[2]):
        for x in range(G.shape[0]):
            for y in range(G.shape[1]):
                if Gm_[x,y] != 0:
                    G[x,y,i] = G[x,y,i]*(Gm[x,y]/Gm_[x,y])
                else:
                    G[x,y,i] = 0
    #get the reconstructed weigh matrix, which is enhanced gradient magnitude
    w_c = 1/(np.abs(Gm)**a+0.0001)
    #Reconstructe the image
    imr = reconstruct_grad_field(G,w_c,im_gray[0,0],im_gray).clip(0,1)
    
    return imr

def enhancedImage_gray2RGB_ratios(im,im_gray,imr):
    """Implement task1(equalising histogram for color images by ratios) to recover colors
    im - The original image
    im_gray - grayscale for the original image
    imr - The reconstructed image
    """
    row,col,channel = im.shape
    #create a array for the recovered image
    new_img = np.zeros((row,col,channel)) 
  
    #restore color from color ratio: Rnew(x; y) = Vnew(x; y)*Rold(x; y)/Vold(x; y)
    for i in range(3):
          for x in range(row):
              for y in range(col):
                  #Compute a normalized cumulative histogram：new_gray_image[x,y] = norm_arr[gray_img[x,y]]
                  new_img[x,y,i] = (imr[x,y]*im[x,y,i])/im_gray[x,y]
      
    new_img = skimage.img_as_float(new_img)

    return new_img

def enhancedImage_gray2RGB_hsv(im,imr):
    """Implement task1(equalising histogram for color images by hsv) to recover colors
    im - The original image
    imr - The reconstructed image
    """
    #convert grayscale image to RGB image

    row,col,channel = im.shape
    hsv_img = color.rgb2hsv(im)
    #create a array for the recovered image
    new_img = np.zeros((row,col,channel)) 
    
    for x in range(row):
          for y in range(col):
              hsv_img[x,y,2] = imr[x,y]
      
    new_img = color.hsv2rgb(hsv_img)

    return new_img
    
        

if __name__ == "__main__":
    #TODO: Replace with your own image
    #im = io.imread(path.join('images','rubberduck.jpg'))
    im = io.imread(path.join('images','dubai.jpg'))
    im = skimage.img_as_float(im)

    im_gray = color.rgb2gray(im)

    #TODO: Implement gradient domain enhancement on the greyscale image, then recover colour

    plt.figure(figsize=(9, 16))

    plt.subplot(121)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(im)

    imr1 = grad_domain_enhancement(im_gray, a = 1)
    plt.subplot(122)
    plt.title('Enhanced(a=1)')
    plt.axis('off')
    imr1_color = enhancedImage_gray2RGB_hsv(im,imr1)
    plt.imshow(imr1_color)
    
    plt.savefig('results/task5.jpg')

    plt.show()
    plt.close
