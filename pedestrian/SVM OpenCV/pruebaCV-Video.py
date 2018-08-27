import cv2
import time
import settings

def main():
    #fig = plt.figure()
    #Inicializacion del HogDescriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    #Inicializacion de captura de video desde camara web o archivo de video
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(settings.vid_path)   
    i = 0
    
    try:    
        while(True):
        
            #Saltea i cantidad de frames
            if i < 2:
                i += 1
                cap.grab()
                continue
            
            i = 0

            #Lee el proximo frame a procesar
            ret, frame = cap.read()
            
            #Pasa el frame a escala de grises y lo reescala
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,None,fx=settings.resize, fy=settings.resize, interpolation = cv2.INTER_CUBIC)
            
            """fig.add_subplot(221)
            plt.title('Test')
            plt.imshow(image)"""	
            
            #Empieza a correr el tiempo para calcular cuanto tarda
            start = time.time()
            
            #Calcula el Hog y hace la deteccion con SVM devolviendo los bounding boxes de los match
            g = HogDescriptor(image,hog)
            
            #Calcula el tiempo transcurrido
            end = time.time()
            parcial = end - start
            print(parcial)
            
            """fig.add_subplot(222)
            plt.title('Test2')
            plt.imshow(cropped_image)	"""

            #print(g)
            #Dibuja los rectangulos en pantalla de lo que detectó
            for (x, y, w, h) in g:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow("Detections", image)
            cv2.waitKey(15)
    finally:
        cv2.destroyWindow("Detections")
    
def HogDescriptor(image,hog):
    (rects, weights) = hog.detectMultiScale(image, winStride=(settings.winStride,settings.winStride),padding=(settings.padding,settings.padding), scale=settings.scaleDetection, useMeanshiftGrouping=False)
    return rects 
        
if __name__ == '__main__':
    main()