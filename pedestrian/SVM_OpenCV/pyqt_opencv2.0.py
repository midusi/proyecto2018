#IS_GAME_ACTIVE = False
RES_WIDTH = 1980
RES_HEIGHT = 1080


import cv2
from PyQt5.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot)
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QHBoxLayout
import sys
import time
import settings
import numpy as np
from datetime import datetime as dt
import skimage.transform
from skimage.feature import hog


class Thread(QThread):
    #changePixmap = pyqtSignal(QImage,QImage) 
    changePixmap = pyqtSignal(QImage) 
    # changePixmap2 = pyqtSignal(QImage)
    # changePixmap3 = pyqtSignal(QImage)

    def run(self):
        winSize = (64, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradients = False
        # hog = cv2.HOGDescriptor()
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Inicializacion de captura de video desde camara web o archivo de video
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,RES_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,RES_HEIGHT)

        i = 0

        oldRect = []
        
        processedHog = None

        f = fps()
        
        lastID = 0



        while True:
            """Consigue la captura"""
            frame, oldRect, visual = getImage(f, i, hog, oldRect, cap, lastID)
            i+=1
            
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            processedFrame = convertToQtFormat.scaled(RES_WIDTH, RES_HEIGHT, Qt.KeepAspectRatio)
            
            """if (visual.any()):
                convertToQtFormat = QImage(visual.data, visual.shape[1], visual.shape[0], QImage.Format_RGB888)
                processedHog = convertToQtFormat.scaled(RES_WIDTH, RES_HEIGHT, Qt.KeepAspectRatio)"""
            
            #self.changePixmap.emit(processedFrame, processedHog)
            self.changePixmap.emit(processedFrame)



class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.title = "Test"

    """@pyqtSlot(QImage, QImage)
    def setImage(self, image, hog):
        self.label.setPixmap(QPixmap.fromImage(image))
        self.labelHog.setPixmap(QPixmap.fromImage(hog))
        #chequear cantidad
            self.labelDetections.setPixmap(QPixmap.fromImage(detections[0]))
            self.labelDetections2.setPixmap(QPixmap.fromImage(detections[1]))"""
            
    
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    # @pyqtSlot(QImage)
    # def setImage2(self, image):
    #     self.label2.setPixmap(QPixmap.fromImage(image))
    
    # @pyqtSlot(QImage)
    # def setImage3(self, image):
    #     self.label3.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle("Test")
        #self.setGeometry(0, 0, RES_WIDTH, RES_HEIGHT)
        #self.setGeometry(0, 0, 1920, 1080)
        self.showFullScreen()
        # self.resize(1800, 1200)
        # create a label
        hbox = QHBoxLayout(self)
        self.label = QLabel(self)
        self.label.move(0, 0)
        self.label.resize(RES_WIDTH-400, RES_HEIGHT)
        
        self.labelHog = QLabel(self)
        self.labelHog.move(RES_WIDTH-400, 0)
        self.labelHog.resize(400, RES_HEIGHT/2)    

        hbox.addWidget(self.label)
        hbox.addWidget(self.labelHog)

        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        # th.changePixmap2.connect(self.setImage2)
        # th.changePixmap3.connect(self.setImage3)
        th.start()

    def changeValue(self):
        val = float(self.slider.value)
        settings.resize = val / 100.0


def HogDescriptor(image, hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride, settings.winStride),
                                            padding=(settings.padding, settings.padding), scale=settings.scaleDetection,
                                            useMeanshiftGrouping=False)
    for item in rects:
        item[2] = int(item[2] // settings.resize)
        item[3] = int(item[3] // settings.resize)
        item[0] = int(item[0] // settings.resize)
        item[1] = int(item[1] // settings.resize)
    
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


def getImage(f, i, hog, oldRect, cap, lastID):
    timeFrame = dt.now()
    ret, frame = cap.read()
    visual = np.array([])
    if ret:
        # Saltea 1 frame
        if i % settings.skip >0:

            # Pasa el frame a escala de grises y lo reescala
            imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(imageGray, None, fx=settings.resize, fy=settings.resize, interpolation=cv2.INTER_CUBIC)


            # Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
            newRect = HogDescriptor(image, hog)
            
            visual = getViewHogs(imageGray)

            oldRect = survivingBBoxes_ms(oldRect, newRect, settings.trackThreshold, timeFrame, lastID)
                
        FPS = f()
        # Dibuja los rectangulos en pantalla de lo que detectó
        for (x, y, w, h, s, id) in oldRect:
            #cv2.rectangle(frame, (int(x // settings.resize), int(y // settings.resize)), (int((x + w) // settings.resize), int((y + h) // settings.resize)), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(round(FPS, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frame, oldRect, visual


def survivingBBoxes_ms(oldRect, newRect, threshold, frameTime, lastID):
    date = dt.now() - frameTime
    if newRect.any():
        newRect = np.pad(newRect, ((0, 0), (0, 2)), 'constant', constant_values=(settings.boundBoxLife))
        for item in oldRect:
            item[4] -= date.microseconds
            iou = overlap(item, newRect)
            if (iou.any()):
                m = max(iou)
                i = np.argmax(iou)
                if (m > threshold):
                    item[0:5] = newRect[i][0:5]
                    newRect = np.vstack([newRect[0:i], newRect[i + 1:]])

        oldRect = list(filter(lambda rect: rect[4] > 0, oldRect))
        for item in newRect:
            item[5] = lastID
            lastID += 1
        bboxes=oldRect + [vbox for vbox in newRect]
        #sizes = [ w*h for (x,y,w,h,s,id) in bboxes]
        bboxes=sorted(bboxes,key= lambda bbox: bbox[2]*bbox[3],reverse=True)
        #sorted_indices=np.argsort(sizes)
        last_index=min(settings.max_bounding_boxes,len(bboxes))
        bboxes=bboxes[0:last_index]
        return bboxes
        #return bboxes

    for item in oldRect:
        item[4] -= 3000
    oldRect = list(filter(lambda rect: rect[4] > 0, oldRect))
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
        self.counter += 1
        if (time.time() - self.start) > 1:
            # print("FPS: ", counter / (time.time() - start))
            self.FPS = self.counter / (time.time() - self.start)
            self.counter = 0
            self.start = time.time()
        return self.FPS
    
    
def getViewHogs(image):
    """Genera el HOG de todas las imagenes que se encuentran
    dentro de la carpeta pasada por parametro"""
    image = skimage.transform.resize(image, (image.shape[0] / 10, image.shape[1] / 10))
    img_hog, visual = hog(image,block_norm='L2-Hys',transform_sqrt=True,visualise=True)
    return visual


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
