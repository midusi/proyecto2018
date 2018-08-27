import cv2
import time
import settings

def main():
    #fig = plt.figure()
    #Inicializacion del HogDescriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    #Inicializacion de captura de video desde camara web o archivo de video
    cap = cv2.VideoCapture(1)
    #cap = cv2.VideoCapture(settings.vid_path)   
    i = 0
    counter = 0
    FPS = 0
    
    start = time.time()
    
    try:    
        while(True):
        
            """#Saltea i cantidad de frames
            if i < 1:
                i += 1
                cap.grab()
                continue
            
            i = 0"""

            #Lee el proximo frame a procesar
            ret, frame = cap.read()
            
            #Pasa el frame a escala de grises y lo reescala
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,None,fx=settings.resize, fy=settings.resize, interpolation = cv2.INTER_CUBIC)
                        
            #Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
            g = HogDescriptor(image,hog)
            
            #Calcula el tiempo transcurrido y muestra FPS
            counter+=1
            if (time.time() - start) > 1 :
                #print("FPS: ", counter / (time.time() - start))                
                FPS = counter / (time.time() - start)
                counter = 0
                start = time.time()
            
            #print(g)
            #Dibuja los rectangulos en pantalla de lo que detectó
            for (x, y, w, h) in g:
                cv2.rectangle(frame, (int(x//settings.resize), int(y//settings.resize)), (int((x + w)//settings.resize), int((y + h)//settings.resize)), (0, 255, 0), 2)
            cv2.putText(frame,str(round(FPS,2)),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.imshow("Detections", frame)
            cv2.waitKey(2)
    finally:
        cv2.destroyWindow("Detections")
    
def HogDescriptor(image,hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=settings.scaleDetection, useMeanshiftGrouping=False)
    return rects 
        
if __name__ == '__main__':
    main()