#Path de la imagen y video para probar deteccion
img_path = 'crop.png'
vid_path = 'video.mp4'
#settings de tracking
trackThreshold = 0.8
boundBoxLife = 150 * 1000  #en microsegundos  1 us = 1000 ms = 1000000
boundBoxFrameLife = 5 #cantidad de frames

#Settings de reescalar
resize = 0.15
boundBoxSize = 0.8

#Settings hogs
resizeHogs = 10 #cuantas veces achicarlo

#Settings de la SVM
#less= better detection, more cpu time
scaleDetection = 2.5

padding = 8
winStride = 4

# skip 1 frame for every 'skip' frames processed
skip=4
assert(skip>1)

# max number of boxes to detect
max_bounding_boxes=5 # set to -1 to use all bounding boxes

colors = [(0,0,255),
          (0,255,0),
          (255,0,0),
          (50,130,255),
          (255,255,0),
          (255,0,255),
          (0,255,255),
          (255,255,255)]

