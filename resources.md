

# Métodos

## Histograms of Oriented Gradients (HOGs)
* [Artículo original](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
* [Tutorial](http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/)
* [skimage API](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog)
* [OpenCV API](https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html)


## Support-Vector Machines (SMVs)
* [Explicación (inglés)](https://www.youtube.com/watch?v=N1vOgolbjSc)
* [Otra explicación (inglés)](https://www.youtube.com/watch?v=eUfvyUEGMD8)
* [Tutorial](http://cs229.stanford.edu/notes/cs229-notes3.pdf)
* [sklearn API](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)

## Non-Maximum Suppresion (NMS)
* [Tutorial](https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)
* [Explicación (ingles)](https://www.youtube.com/watch?v=A46HZGR5fMw)

## Papers relevantes

## Personas y peatones
* [Pedestrian Detection:
An Evaluation of the State of the Art](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5975165)
* [How Far are We from Solving Pedestrian Detection?](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_How_Far_Are_CVPR_2016_paper.pdf)
* [Is Faster R-CNN Doing Well for
Pedestrian Detection?](http://kaiminghe.com/publications/eccv16ped.pdf) (Faster RCNN es uno de los dos mejores métodos de detección actuales, junto con YOLO)

## Autos

# Bases de datos

## Personas y peatones
* [Daimler pedestrian](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Mono_Ped__Detection_Be/daimler_mono_ped__detection_be.html)  8.5gb, grayscale,640x480, 15500 positive training examples, 6744 negative training images for cropping, 22000 testing images with 55000 pedestrian labels
* [Caltech pedestrian](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) 11gb, 640x480, 250.000 frames, 350000 bbs, 2300 pedestrians
* [INRIA person](http://pascal.inrialpes.fr/data/human/) 1gb

## Autos

* [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 16,185 imágenes de 196 clases de autos, (~8000 para train, ~8000 para split). En este caso no nos interesan las clases de autos sino las bbs.
* [TME Motorway](http://cmp.felk.cvut.cz/data/motorway/) ~15gb, 1024x768, 20hz, yo tengo el .gpg para desencriptar
