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
    height, width, channels = imageGray.shape

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
    
    l = g.copy()
    
    recorte = [width, height, 0, 0]
    
    for item in l:
        item[0] = item[0] - item[2]
        item[2] = item[2]*3
        item[1] = item[1]-(0.1*item[3])
        item[3] = 1.1*item[3]
        if(item[0]<0):
            item[0]=0  
        if(item[1]<0):
            item[1]=0
        if(item[3]>height-item[1]):
            item[3] = height-item[1]-1
        if(item[2]>width-item[0]):
            item[2] = width-item[0]-1
        print(item)
        if(recorte[0]>item[0]):
            recorte[0] = item[0]
        if(recorte[1]>item[1]):
            recorte[1] = item[1]
            
    for item in l:        
        if(recorte[2]<item[2]+item[0]-recorte[0]):
            recorte[2] = item[2]+item[0]-recorte[0]
        if(recorte[3]<item[3]+item[1]-recorte[1]):
            recorte[3] = item[3]+item[1]-recorte[1]
            
    print(recorte)
        
    #Dibuja los rectangulos en pantalla de lo que detectó
    for (x, y, w, h) in g:
        cv2.rectangle(image, (int(x//settings.resize), int(y//settings.resize)), (int((x + w)//settings.resize), int((y + h)//settings.resize)), (0, 255, 0), 2)
    #cv2.rectangle(image, (int(recorte[0]//settings.resize), int(recorte[1]//settings.resize)), (int((recorte[0] + recorte[2])//settings.resize), int((recorte[1] + recorte[3])//settings.resize)), (0, 0, 255), 2)
    
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Detections")
    
    croppedimageGray = imageGray[recorte[1]:recorte[1]+recorte[3], recorte[0]:recorte[0]+recorte[2]]
    
    g2 = HogDescriptor(croppedimageGray,hog)
    
    for (x, y, w, h) in g2:
        cv2.rectangle(image, (int((x+recorte[0])//settings.resize), int((y+recorte[1])//settings.resize)), (int(((x+recorte[0]) + w)//settings.resize), int(((y+recorte[1]) + h)//settings.resize)), (0, 0, 255), 2)
    
    
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Detections")
    
def HogDescriptor(image,hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=settings.scaleDetection, useMeanshiftGrouping=False)
    return rects 
        
if __name__ == '__main__':
    main()