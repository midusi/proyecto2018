import cv2
import time
import settings

def main():
    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 2
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
        
    #Carga la imagen en escala de grises y la reescala
    image = cv2.imread(settings.img_path)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGray = cv2.resize(image,None,fx=settings.resize, fy=settings.resize, interpolation = cv2.INTER_CUBIC)

    """#Empieza a correr el tiempo para calcular cuanto tarda
    print("Time started ...")
    start = time.time()"""
    
    #Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
    g = HogDescriptor(imageGray,hog)
    
    """#Calcula el tiempo transcurrido
    end = time.time()
    print("Time Finished ...")
    parcial = end - start
    print(parcial)"""
        
    #print(g)
    #Dibuja los rectangulos en pantalla de lo que detectó
    for (x, y, w, h) in g:
        cv2.rectangle(image, (int(x//settings.resize), int(y//settings.resize)), (int((x + w)//settings.resize), int((y + h)//settings.resize)), (0, 255, 0), 2)
    
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Detections")
    
def HogDescriptor(image,hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=settings.scaleDetection, useMeanshiftGrouping=False)
    return rects 
        
if __name__ == '__main__':
    main()