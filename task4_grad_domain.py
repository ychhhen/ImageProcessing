#Task 4: Gradient domain reconstruction
import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
#from sksparse.cholmod import cholesky
#from scikits.sparse.cholmod import cholesky
from scipy import signal
import skimage
import time

def img2grad_field(img):
    """Return a gradient field for a greyscale image
    The function returns image [height,width,2], where the last dimension selects partial derivates along x or y
    """
    # img must be a greyscale image
    sz = img.shape
    G = np.zeros([sz[0], sz[1], 2])
    # Gradients along x-axis
    G[:,:,0] = signal.convolve2d( img, np.array([1, -1, 0]).reshape(1,3), 'same', boundary='symm' )
    # Gradients along y-axis
    G[:,:,1] = signal.convolve2d( img,  np.array([1, -1, 0]).reshape(3,1), 'same', boundary='symm' )
    return G

def reconstruct_grad_field( G, w, v_00, img ):
    """Reconstruct a (greyscale) image from a gradcient field
    G - gradient field, for example created with img2grad_field
    w - weight assigned to each gradient
    v_00 - the value of the first pixel 
    """
    sz = G.shape[:2]  
    N = sz[0]*sz[1] #number of pixels

    # Gradient operators as sparse matrices
    o1 =  np.ones((N,1))
    B = np.concatenate( (-o1, np.concatenate( (np.zeros((sz[0],1)), o1[:N-sz[0]]), 0 ) ), 1)
    B[N-sz[0]:N,0] = 0
    Ogx = sparse.spdiags(B.transpose(), [0 ,sz[0]], N, N ) # Forward difference operator along x

    B = np.concatenate( (-o1 ,np.concatenate((np.array([[0]]), o1[0:N-1]) ,0)), 1)
    B[sz[0]-1::sz[0], 0] = 0
    B[sz[0]::sz[0],1] = 0
    Ogy = sparse.spdiags(B.transpose(), [0, 1], N, N ) # Forward difference operator along y

    #TODO: Implement the gradient domain reconstruction 
    
    #define some parameters
    #transpose Ogx and Ogy
    Ogx_t = Ogx.T
    Ogy_t = Ogy.T
    
    #G
    G_x = sparse.csr_matrix(G[:,:,0].flatten(order='F')).T
    G_y = sparse.csr_matrix(G[:,:,1].flatten(order='F')).T
    
    #define sparse diagonal matrix W
    W = sparse.spdiags(w.flatten(order='F'),0,N,N)
    
    #define C
    C = np.zeros(N)
    C[0] = 1 
    C_ = sparse.csr_matrix(C.reshape(1,N))
    C_t = C_.T

#    build A and b
    A = Ogx_t.dot(W).dot(Ogx) + Ogy_t.dot(W).dot(Ogy) + C_t.dot(C_)
    b = Ogx_t.dot(W).dot(G_x) + Ogy_t.dot(W).dot(G_y) + C_t*v_00
    #compute in sparse.linalg
    start = time.time()
    x_0 = linalg.spsolve(A,b)
    end = time.time()
    imr = x_0.reshape((sz[0],sz[1]),order = 'F')
    imr = skimage.img_as_float(imr)
    
#    #compute in cholesky
#    start = time.time()
#    factor = cholesky(A)
#    x_1 = factor(b)
#    end = time.time()
#    imr = x_1.reshape(sz[0],sz[1],order = 'F')
#    imr = skimage.img_as_float(imr)
    
    #show each running time
    print(str(end-start))
    
    return imr


if __name__ == "__main__":
    #TODO: Replace with your own image
    im = io.imread(path.join('images','people.jpg'), as_gray=True)
    im = skimage.img_as_float(im)

    G = img2grad_field(im)
    Gm = np.sqrt(np.sum(G*G, axis=2))
    
    #w = np.ones (img.shape)
    w = 1/(Gm + 0.0001)     # To avoid pinching artefacts

    imr = reconstruct_grad_field(G,w,im[0,0],im).clip(0,1)
    #print(imr)

    plt.figure(figsize=(12, 7))

    plt.subplot(131)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(im,cmap='gray')

    plt.subplot(132)
    plt.title('Reconstructed')
    plt.axis('off')
    plt.imshow(imr,cmap='gray')

    plt.subplot(133)
    plt.title('Difference')
    plt.axis('off')
    plt.imshow(imr-im)
    
    plt.savefig('result/task4.jpg')
    plt.show()
