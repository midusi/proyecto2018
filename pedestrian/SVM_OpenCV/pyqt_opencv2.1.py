#IS_GAME_ACTIVE = False
RES_WIDTH = 1920
RES_HEIGHT = 1080


import cv2
from PyQt5.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
from PyQt5.QtGui import (QPixmap, QImage, QFont)
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QHBoxLayout,  QGridLayout
import sys
import time
import settings
import numpy as np
from datetime import datetime as dt
import skimage.transform
from skimage.feature import hog
from skimage import exposure
from matplotlib import pyplot as plt
from qimage2ndarray import array2qimage


class Thread(QThread):
    changePixmap = pyqtSignal(QImage,QImage, object)

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
        # cap = cv2.VideoCapture("video3.mp4")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,RES_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,RES_HEIGHT)

        i = 0

        oldRect = []
        
        newHog = None
        detectionsHogs = []
        

        f = fps()
        
        lastID = 0



        while True:
            """Consigue la captura"""
            frame, oldRect, lastID, visual, detectionsHogs = getImage(f, i, hog, oldRect, cap, lastID)
            i+=1
            
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            processedFrame = convertToQtFormat.scaled(RES_WIDTH-652, RES_HEIGHT, Qt.KeepAspectRatio)
            
            if (visual.any()):
                newHog = visual
                        
            if newHog is not None:   
                convertToQtFormat = QImage(newHog.data, newHog.shape[1], newHog.shape[0], QImage.Format_RGBA8888)
                processedHog = convertToQtFormat.scaled(652, RES_HEIGHT/2, Qt.KeepAspectRatio)
            else:
                processedHog = processedFrame
            self.changePixmap.emit(processedFrame, processedHog, detectionsHogs)



class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.title = "Test"

    @pyqtSlot(QImage, QImage, object)
    def setImage(self, image, hog, detections):
        self.label.setPixmap(QPixmap.fromImage(image))
        self.labelHog.setPixmap(QPixmap.fromImage(hog.mirrored(vertical=False, horizontal=True)))
        if len(detections)>0:
            detection_0 = array2qimage(detections[0])
            detection_0 = detection_0.mirrored(vertical=False, horizontal=True)
            detection_0 = detection_0.scaled(RES_WIDTH/2, RES_HEIGHT/2, Qt.KeepAspectRatio)
            detection_0 = QPixmap.fromImage(detection_0)
            self.labelDetection.setPixmap(detection_0)
            self.resetTime(1)
            if len(detections)>1:                
                detection_1 = array2qimage(detections[1])
                detection_1 = detection_1.mirrored(vertical=False, horizontal=True)
                detection_1 = detection_1.scaled(RES_WIDTH/2, RES_HEIGHT/2, Qt.KeepAspectRatio)
                detection_1 = QPixmap.fromImage(detection_1)
                self.labelDetection2.setPixmap(detection_1)
                self.resetTime(2)
            

    def initUI(self):
        self.setWindowTitle("Test")
        
        self.showFullScreen()
        
        self.setStyleSheet("background-color: black;")
        
        # create a label
        self.label = QLabel(self)
        self.label.resize(RES_WIDTH-652, RES_HEIGHT)
        self.label.move(0, 0)
        
        self.labelHog = QLabel(self)
        self.labelHog.move(RES_WIDTH-652, 0)
        self.labelHog.resize(652, RES_HEIGHT/2)    
        
        self.labelDetection = QLabel(self)
        self.labelDetection.move(RES_WIDTH-600, RES_HEIGHT/2)
        self.labelDetection.resize(326, RES_HEIGHT/2)   
        
        self.labelDetection2 = QLabel(self)
        self.labelDetection2.move(RES_WIDTH-300, RES_HEIGHT/2)
        self.labelDetection2.resize(326, RES_HEIGHT/2)   
        
        newfont = QFont("Times", 36, QFont.Bold)
        
        self.labelText = QLabel(self)
        self.labelText.setText("Vista Principal")
        self.labelText.setStyleSheet('color: yellow')
        self.labelText.setFont(newfont)
        self.labelText.move(450,80)
        
        self.labelHogText = QLabel(self)
        self.labelHogText.setText("Vista Hogs")
        self.labelHogText.setStyleSheet('color: yellow')
        self.labelHogText.setFont(newfont)
        self.labelHogText.move(RES_WIDTH-450,10)
        
        self.labelDetectionsText = QLabel(self)
        self.labelDetectionsText.setText("Detecciones 1º y 2º")
        self.labelDetectionsText.setStyleSheet('color: yellow')
        self.labelDetectionsText.setFont(newfont)
        self.labelDetectionsText.move(RES_WIDTH-550,(RES_HEIGHT/2)-75)
        
        
        self.label.show()
        self.labelHog.show()
        self.labelDetection.show()
        self.labelDetection2.show()
        self.labelText.show()
        self.labelHogText.show()
        self.labelDetectionsText.show()
        
        
        self.time_to_wait = 2
        self.time_to_wait2 = 2       
        
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.detectionLife)
        self.timer.start()

        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    def changeValue(self):
        val = float(self.slider.value)
        settings.resize = val / 100.0
    
    def detectionLife(self):
        self.time_to_wait -= 1
        self.time_to_wait2 -= 1
        if self.time_to_wait <= 0:
            self.labelDetection.clear()        
        if self.time_to_wait2 <= 0:
            self.labelDetection2.clear()
    
    def resetTime(self,id):
        if id == 1:
            self.time_to_wait = 2
        if id == 2:
            self.time_to_wait2 = 2


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
    hog_detection = []
    if ret:
        # Saltea 1 frame
        if i % settings.skip >0:

            # Pasa el frame a escala de grises y lo reescala
            imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(imageGray, None, fx=settings.resize, fy=settings.resize, interpolation=cv2.INTER_CUBIC)


            # Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
            newRect = HogDescriptor(image, hog)
            
            visual = getViewHogs(imageGray,settings.resizeHogs)

            oldRect, lastID = survivingBBoxes_ms(oldRect, newRect, settings.trackThreshold, timeFrame, lastID)
                
        FPS = f()
        # Dibuja los rectangulos en pantalla de lo que detectó
        for (x, y, w, h, s, id) in oldRect:
            #cv2.rectangle(frame, (int(x // settings.resize), int(y // settings.resize)), (int((x + w) // settings.resize), int((y + h) // settings.resize)), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), settings.colors[id%8], 2)
            
        # Recorta los hogs de las primeras 2 detecciones
        if visual.any() and len(oldRect) > 0:
            #hog_detection.append(visual[int(oldRect[0][1]//settings.resizeHogs):int((oldRect[0][1]+oldRect[0][3])//settings.resizeHogs), int(oldRect[0][0]//settings.resizeHogs):int((oldRect[0][0]+oldRect[0][2])//settings.resizeHogs)])
            hog_detection.append(getViewHogs(image[int(oldRect[0][1]):int((oldRect[0][1]+oldRect[0][3])), int(oldRect[0][0]):int((oldRect[0][0]+oldRect[0][2]))],1))
            if len(oldRect) > 1:
                hog_detection.append(getViewHogs(image[int(oldRect[1][1]):int((oldRect[1][1]+oldRect[1][3])), int(oldRect[1][0]):int((oldRect[1][0]+oldRect[1][2]))],1))
                #hog_detection.append(visual[int(oldRect[1][1]//settings.resizeHogs):int((oldRect[1][1]+oldRect[1][3])//settings.resizeHogs), int(oldRect[1][0]//settings.resizeHogs):int((oldRect[1][0]+oldRect[1][2])//settings.resizeHogs)])
        
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(round(FPS, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frame, oldRect, lastID, visual, hog_detection


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
        return bboxes, lastID
        #return bboxes

    for item in oldRect:
        item[4] -= 3000
    oldRect = list(filter(lambda rect: rect[4] > 0, oldRect))
    return oldRect, lastID


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
    
    
def getViewHogs(image,size):
    """Genera el HOG de todas las imagenes que se encuentran
    dentro de la carpeta pasada por parametro"""    
    #image = image[...,::-1]
    if size != 1:
        image = skimage.transform.resize(image, (image.shape[0] / size, image.shape[1] / size))
    img_hog, visual = hog(image,block_norm='L2-Hys',transform_sqrt=True,visualise=True)
    visual = exposure.rescale_intensity(visual)    
    norm = plt.Normalize(vmin=visual.min(), vmax=visual.max())
    visual = plt.cm.jet(norm(visual))
    visual = 255*visual
    visual = visual.astype(np.uint8)    
    return visual


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
