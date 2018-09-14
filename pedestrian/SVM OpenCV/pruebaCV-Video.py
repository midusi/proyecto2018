import cv2
import time
import settings
import numpy as np
from datetime import datetime as dt


def main():
    #fig = plt.figure()
    #Inicializacion del HogDescriptor
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
    #hog = cv2.HOGDescriptor()
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    #Inicializacion de captura de video desde camara web o archivo de video
    cap = cv2.VideoCapture("video3.mp4")
    #cap = cv2.VideoCapture(settings.vid_path)   
    i = True
    counter = 0
    FPS = 0
    oldRect = []
    
    start = time.time()
    
    try:    
        while(True):
            timeFrame = dt.now()

            #Lee el proximo frame a procesar
            ret, frame = cap.read()
            
            #Saltea 1 frame
            if i == True:
                i = False
                #Pasa el frame a escala de grises y lo reescala
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image,None,fx=settings.resize, fy=settings.resize, interpolation = cv2.INTER_CUBIC)
                            
                #Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
                newRect = HogDescriptor(image,hog)
                
                oldRect = survivingBBoxes_ms(oldRect, newRect, settings.trackThreshold, timeFrame)
            else:
                i = True
            
            #Calcula el tiempo transcurrido y muestra FPS
            counter+=1
            if (time.time() - start) > 1 :
                #print("FPS: ", counter / (time.time() - start))                
                FPS = counter / (time.time() - start)
                counter = 0
                start = time.time()
            
            #Dibuja los rectangulos en pantalla de lo que detectó
            for (x, y, w, h, s) in oldRect:
                cv2.rectangle(frame, (int(x//settings.resize), int(y//settings.resize)), (int((x + w)//settings.resize), int((y + h)//settings.resize)), (0, 255, 0), 2)
            cv2.putText(frame,str(round(FPS,2)),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.imshow("Detections", frame)
            cv2.waitKey(2)
    finally:
        cv2.destroyWindow("Detections")
        
def HogDescriptor(image,hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=settings.scaleDetection, useMeanshiftGrouping=False)
       
    return np.array(rects)

def overlap(box, boxes):
    ww = np.maximum(np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) -
                    np.maximum(box[0], boxes[:, 0]),
                    0)
    hh = np.maximum(np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) -
                    np.maximum(box[1], boxes[:, 1]),
                    0)
    uu = box[2] * box[3] + boxes[:, 2] * boxes[:, 3]
    return ww * hh / (uu - ww * hh)

def survivingBBoxes_ms(oldRect, newRect, threshold, frameTime):
    date = dt.now()-frameTime
    if newRect.any():        
        newRect = np.pad(newRect,((0,0),(0,1)), 'constant', constant_values=(settings.boundBoxLife))
        for item in oldRect:
            item[4] -= date.microseconds
            iou = overlap(item,newRect)
            if(iou.any()):
                m = max(iou)
                i = np.argmax(iou)
                if(m>threshold):
                    item[0:5] = newRect[i]
                    newRect = np.vstack([newRect[0:i] , newRect[i+1:]])
            
        oldRect = list(filter(lambda rect: rect[4]>0 , oldRect))
        return oldRect + [vbox for vbox in newRect]
    
    for item in oldRect:
        item[4] -= 3000
    oldRect = list(filter(lambda rect: rect[4]>0 , oldRect))
    return oldRect

#def fps():
    

if __name__ == '__main__':
    main()