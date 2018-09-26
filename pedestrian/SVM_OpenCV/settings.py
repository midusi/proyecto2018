# Settings de reescalar
resize = 0.3  # Epsilon de cuanto se achica la imagen original en porcentaje

# Settings de la SVM
scaleDetection = 1.5  # Epsilon de la piramide
padding = 16  # Padding con el que se
winStride = 5  # Stride
winHeight = 400  # Alto de la ventana
winWidth = 200  # Ancho de la ventana
scoreThreshold = 5  # Score minimo que debe tener para ser tomado como peaton

# Video
countIgnoredFrames = 3  # Cantidad de frames que se ignoran antes de leer uno nuevo

# Tracking
trackThreshold = 0.6
boundBoxLife = 50000
