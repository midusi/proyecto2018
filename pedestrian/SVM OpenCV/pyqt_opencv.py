import cv2
from PyQt5.QtCore import (QThread, Qt, pyqtSignal,pyqtSlot)
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtWidgets import QWidget,QLabel,QApplication, QSlider, QHBoxLayout, QLineEdit
import sys
import time
import settings
import numpy as np
from datetime import datetime as dt

class Thread(QThread):


    changePixmap = pyqtSignal(QImage)
    changePixmap2 = pyqtSignal(QImage)
    changePixmap3 = pyqtSignal(QImage)

    def run(self):
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
        cap = cv2.VideoCapture(0) 
        #cap2 = cv2.VideoCapture(1) 
        i = True
        i2 = True
        oldRect = []
        oldRect2 = []
        
        f = fps()    
        f2 = fps()    
        
        while True:
                """Consigue la captura"""
                frame, oldRect, i = getImage(f, i, hog, oldRect, cap)
                frame2, oldRect2, i2 = getImage(f2, i2, hog, oldRect2, cap)
                
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1440, 1080, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                
                rgbImage2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                convertToQtFormat2 = QImage(rgbImage2.data, rgbImage2.shape[1], rgbImage2.shape[0], QImage.Format_RGB888)
                p2 = convertToQtFormat2.scaled(512, 384, Qt.KeepAspectRatio)
                #self.changePixmap2.emit(p)
                #self.changePixmap3.emit(p)



class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.title = "Test"

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
        
    @pyqtSlot(QImage)
    def setImage2(self, image):
        self.label2.setPixmap(QPixmap.fromImage(image))
        
    @pyqtSlot(QImage)
    def setImage3(self, image):
        self.label3.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle("Test")
        self.setGeometry(0,0, 1920,1080)
        self.showFullScreen()
        #self.resize(1800, 1200)
        # create a label
        hbox = QHBoxLayout(self)
        self.label = QLabel(self)
        self.label.move(0, 0)
        self.label.resize(1440, 1080)
        self.label2 = QLabel(self)
        self.label2.move(1440, 0)
        self.label2.resize(512, 384)
        self.label3 = QLabel(self)
        self.label3.move(1440, 384)
        self.label3.resize(512, 384)
        hbox.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setTickInterval(10)
        self.slider.setGeometry(300, 20, 100, 30)
        self.slider.setMinimum(20)
        self.slider.setMaximum(100)
        self.slider.setValue(40)
        self.slider.valueChanged.connect(self.changeValue)
                
        """self.qle = QLineEdit(self)
        self.qle.editingFinished.connect(self.onChanged)"""
        
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.changePixmap2.connect(self.setImage2)    
        th.changePixmap3.connect(self.setImage3)       
        th.start()
        
    def changeValue(self):
        val = float(self.slider.value)
        settings.resize = val/100.0
        
        
        
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

class fps():
    start = 0
    counter = 0
    FPS = 0
    def __init__(self):
        self.FPS = 0
        self.counter = 0
        self.start = time.time()
        
    def __call__(self):
        self.counter+=1
        if (time.time() - self.start) > 1 :
            #print("FPS: ", counter / (time.time() - start))                
            self.FPS = self.counter / (time.time() - self.start)
            self.counter = 0
            self.start = time.time()
        return self.FPS
        
def getImage(f,i, hog, oldRect, cap):
    timeFrame = dt.now()
    ret, frame = cap.read()
    if ret:
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
        
        FPS = f()
        
        frame = cv2.flip(frame,1)
        
        #Dibuja los rectangulos en pantalla de lo que detectó
        for (x, y, w, h, s) in oldRect:
            #cv2.rectangle(frame, (int(x//settings.resize), int(y//settings.resize)), (int(((x + w)*settings.boundBoxSize)//settings.resize), int(((y + h)*settings.boundBoxSize)//settings.resize)), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x//settings.resize), int(y//settings.resize)), (int((x + w)//settings.resize), int((y + h)//settings.resize)), (0, 255, 0), 2)
        cv2.putText(frame,str(round(FPS,2)),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    return frame, oldRect, i


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())