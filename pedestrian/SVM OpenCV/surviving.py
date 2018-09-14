import cv2
import settings
import numpy as np

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
    
    oldRect = []
    c = 3
    while(c>0):   
        c -= 1
        #Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
        newRect = HogDescriptor(imageGray,hog)
        
        oldRect = survivingBBoxes(oldRect, newRect, settings.trackThreshold)  
        
        print(oldRect)
        
        #Dibuja los rectangulos en pantalla de lo que detectó
        for (x, y, w, h, s) in oldRect:
            cv2.rectangle(image, (int(x//settings.resize), int(y//settings.resize)), (int((x + w)//settings.resize), int((y + h)//settings.resize)), (0, 255, 0), 2)
        #cv2.rectangle(image, (int(recorte[0]//settings.resize), int(recorte[1]//settings.resize)), (int((recorte[0] + recorte[2])//settings.resize), int((recorte[1] + recorte[3])//settings.resize)), (0, 0, 255), 2)
        
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Detections")
    
def HogDescriptor(image,hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=settings.scaleDetection, useMeanshiftGrouping=False)
    if len(rects):
        return np.pad(rects,((0,0),(0,1)), 'constant', constant_values=(settings.boundBoxFrameLife))
    else:        
        return rects

def overlap(box, boxes):
    ww = np.maximum(np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) -
                    np.maximum(box[0], boxes[:, 0]),
                    0)
    hh = np.maximum(np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) -
                    np.maximum(box[1], boxes[:, 1]),
                    0)
    uu = box[2] * box[3] + boxes[:, 2] * boxes[:, 3]
    return ww * hh / (uu - ww * hh)

def survivingBBoxes(oldRect, newRect, threshold):
    if len(newRect):
        for item in oldRect:
            item[4] -= 1
            iou = overlap(item,newRect)
            m = max(iou)
            i = np.argmax(iou)
            if(m>threshold):
                item[0:5] = newRect[i]
                newRect = np.vstack([newRect[0:i] , newRect[i+1:]])
            
        oldRect = list(filter(lambda rect: rect[4]>0 , oldRect))
        return oldRect + [vbox for vbox in newRect]
    
    for item in oldRect:
        item[4] -= 1
    oldRect = list(filter(lambda rect: rect[4]>0 , oldRect))
    return oldRect
        
if __name__ == '__main__':
    main()