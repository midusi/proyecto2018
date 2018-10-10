# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:24:35 2018

@author: busca
"""
import skimage.io
import skimage.transform
from skimage.feature import hog
from skimage import exposure
import cv2
from matplotlib import pyplot as plt

finalSize = [96, 48]

def viewHogs(image):
    """Genera el HOG de todas las imagenes que se encuentran
    dentro de la carpeta pasada por parametro"""
    image = skimage.transform.resize(image, (image.shape[0] / 10, image.shape[1] / 10))
    img_hog, visual = hog(image,block_norm='L2-Hys',transform_sqrt=True,visualise=True)
    return visual
    

def main():
    #Carga la imagen en escala de grises y la reescala
    cap = cv2.VideoCapture(0) 
    i = True
    
    try:    
        while(True):
            #timeFrame = dt.now()

            #Lee el proximo frame a procesar
            ret, frame = cap.read()
            
            #Saltea 1 frame
            if i == True:
                i = False
                #Pasa el frame a escala de grises y lo reescala
                imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                visual = viewHogs(imageGray)	
                visual = exposure.rescale_intensity(visual, in_range=(0, 10))
                
                print(visual)
                
            else:
                i = True            
            
            cv2.imshow("Detections", visual)
            cv2.waitKey(2)
    finally:
        cv2.destroyWindow("Detections")    

if __name__ == '__main__':
    main()	