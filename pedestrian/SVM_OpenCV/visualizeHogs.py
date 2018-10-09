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
    img_hog, visual = viewHogs(imageGray)	
    print(visual)
    
    end = time.time()
    print("Time Finished ...")
    parcial = end - start
    print(parcial)
    
    fig,(ax_ori, ax_image)=plt.subplots(1,2,dpi=500)
    ax_ori.imshow(image)
    ax_image.imshow(visual)
    plt.tight_layout()
    plt.show()	 # Devuelve lista de hogs y el tamaño total de elementos generados

if __name__ == '__main__':
    main()	