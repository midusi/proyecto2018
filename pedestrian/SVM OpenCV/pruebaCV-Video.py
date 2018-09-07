import cv2
import time
import settings
import numpy as np

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
    cap = cv2.VideoCapture(1)
    #cap = cv2.VideoCapture(settings.vid_path)   
    i = True
    counter = 0
    FPS = 0
    
    start = time.time()
    
    try:    
        while(True):
        
            """#Saltea 1 frame
            if i == True:
                i = False
                counter+=1
                cap.grab()
                continue
            i = True"""

            #Lee el proximo frame a procesar
            ret, frame = cap.read()
            
            #Saltea 1 frame
            if i == True:
                i = False
                #Pasa el frame a escala de grises y lo reescala
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image,None,fx=settings.resize, fy=settings.resize, interpolation = cv2.INTER_CUBIC)
                            
                #Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
                g,w = HogDescriptor(image,hog)
                ng = non_max_suppression(g, overlapThresh=0.65)
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
            for (x, y, w, h) in ng:
                cv2.rectangle(frame, (int(x//settings.resize), int(y//settings.resize)), (int((x + w)//settings.resize), int((y + h)//settings.resize)), (0, 255, 0), 2)
            cv2.putText(frame,str(round(FPS,2)),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.imshow("Detections", frame)
            cv2.waitKey(2)
    finally:
        cv2.destroyWindow("Detections")
        
        
def non_max_suppression(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
        
def HogDescriptor(image,hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=settings.scaleDetection)
    return rects, weights
        
if __name__ == '__main__':
    main()