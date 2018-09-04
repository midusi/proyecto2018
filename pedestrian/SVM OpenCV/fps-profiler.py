import pylab as pl     
import cv2
import time
import settings

def main():
    skip = False
    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = False
    #Inicializacion del HogDescriptor
    #hog = cv2.HOGDescriptor()
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    s = ""
    for i in pl.frange(1.05,1.5,0.05):
        s += ';'
        s += str(round(i,2))
    s += '\n'        
        
    #Carga la imagen en escala de grises y la reescala
    image = cv2.imread(settings.img_path, cv2.IMREAD_GRAYSCALE)
    
    for i in pl.frange(0.3,0.8,0.05):
        s += str(round(i,2))
        for j in pl.frange(1.05,1.5,0.05):
            counter = 0
            start = time.time()
            #for k in range(0,30):
            while((time.time() - start) < 1):
                #Saltea 1 frame
                if skip == False:        
                    imagetemp = cv2.resize(image,None,fx=i, fy=i, interpolation = cv2.INTER_CUBIC)
                    
                    #Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
                    g = HogDescriptor(imagetemp,hog,j)
                skip = not skip
                counter += 1
                
            t = time.time() - start
            #FPS = t/30
            FPS = counter / (time.time() - start)         
            s += ";"
            s += str(round(FPS,2))
        s += '\n'
    f = open('csvfile2.txt','w')
    f.write(s.replace('.',','))
    f.close()
    
def HogDescriptor(image,hog,i):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=i)
    return rects 
        
if __name__ == '__main__':
    main()
    
