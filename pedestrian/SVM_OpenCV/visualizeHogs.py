# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:24:35 2018

@author: busca
"""
import skimage.io
import skimage.transform
from skimage.feature import hog
import cv2
from matplotlib import pyplot as plt
import time
import settings
import numpy

finalSize = [96, 48]

def viewHogs(image):
    """Genera el HOG de todas las imagenes que se encuentran
    dentro de la carpeta pasada por parametro"""
    image = skimage.transform.resize(image, (image.shape[0] / 10, image.shape[1] / 10))
    img_hog, visual = hog(image,block_norm='L2-Hys',transform_sqrt=True,visualise=True)
    return visual
    

def main():
    #Carga la imagen en escala de grises y la reescala
    image = cv2.imread(settings.img_path)
    
    print("Time started ...")
    start = time.time()
    
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visual = viewHogs(imageGray)	
    
    end = time.time()
    print("Time Finished ...")
    parcial = end - start
    print(parcial)
    
    
    norm = plt.Normalize(vmin=visual.min(), vmax=visual.max())
    image = plt.cm.jet(norm(visual))
    
    image = 255*image
    image = image.astype(numpy.uint8)
    
    print(image)
    
    fig,(ax_ori, ax_image)=plt.subplots(1,2,dpi=500)
    ax_ori.imshow(image)
    ax_image.imshow(visual)
    
    data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    #plt.imsave('test.png', image)
    
    """fig,(ax_ori, ax_image)=plt.subplots(1,2,dpi=500)
    ax_ori.imshow(image)
    ax_image.imshow(visual)
    plt.tight_layout()
    plt.show()	 # Devuelve lista de hogs y el tamaño total de elementos generados"""
    
    
    def fig2data ( fig ):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )
     
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
        buf.shape = ( w, h,4 )
     
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = numpy.roll ( buf, 3, axis = 2 )
        return buf
    
    
    import Image
 
    def fig2img ( fig ):
        """
        @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
        @param fig a matplotlib figure
        @return a Python Imaging Library ( PIL ) image
        """
        # put the figure pixmap into a numpy array
        buf = fig2data ( fig )
        w, h, d = buf.shape
        return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )

if __name__ == '__main__':
    main()	