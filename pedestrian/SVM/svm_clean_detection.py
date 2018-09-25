import utils
import cv2
import time
import sys
import settings

#VISUALIZE_SLIDDING_WINDOW = False  # Si esta en True se muestra la ventana deslizante en tiempo real

#EPSILON = 1.5

def main():
    """La mayoria del codigo se saco de
    https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/"""
    # Tamaño de ventana deslizante
    #(win_w, win_h) = (200, 400)
    
    # Cargo y preparo la imagen
    # image = utils.load_image_from_path('/home/genaro/PycharmProjects/proyecto2018/pedestrian/SVM/imgs/street4.jpg')
    image = utils.load_image_from_path(settings.img_path)
    #print(image.shape)

    cv2.startWindowThread()
    cv2.namedWindow("Window final")
    #cv2.imshow("Window final", image)
    #cv2.waitKey(0)
    
    image = utils.to_grayscale(image)        
    image = utils.normalize_image_max(image)
    
    count = 0
    total = 0

    # final_bounding_boxes = np.array([], dtype='int16').reshape(0, 4)
    final_bounding_boxes = []
    
    # Obtengo mi SVM
    svm = utils.load_checkpoint(settings.checkpoint_path)
    
    print("Time started ...")
    start = time.time()
    # Loop sobre las diferentes escalas
    for i, resized_image in enumerate(utils.get_pyramid(image, scale=settings.EPSILON)):
        coefficient = settings.EPSILON ** i
        #clone = resized_image.copy()  # Hago una copia de la imagen original para graficar detecciones
        # clone_suppression = clone.copy()  # Hago una copia para graficar el resultado de NMS

        # Lista de bounding boxes para el Non Maximal Suppression
        # bounding_boxes = []

        # Llevo control para saber si funciona el NMS
        # count_bounding_boxes = 0

        # Loop sobre la ventana deslizante en diferentes posiciones
        for (x, y, window) in utils.get_sliding_window(resized_image, stepSize=(32, 64), windowSize=(
        settings.win_w, settings.win_h)):
            # Si la ventana no coincide con nuestro tamaño de ventana, se ignora
            if window.shape[0] != settings.win_h or window.shape[1] != settings.win_w:
                continue

            # Manejo la ventana deslizante actual
            # cropped_image = utils.crop_image(image, x, y, win_w, win_h)  # Recorto la ventana
            # cropped_image = window
            
            cropped_image = utils.resize(window, [96, 48])  # Escalo
            
            startParcial = time.time()

            cropped_image_hog = utils.get_hog_from_image(cropped_image, normalize=False)  # Obtengo el HOG
                        
            end = time.time()
            parcial = end - startParcial
            print(parcial)
            count = count + 1
            total = total + parcial
            #sys.exit()

            # Comienzo a predecir
            prediction = svm.predict([cropped_image_hog])[0]
            if prediction == 1:
                # Si es un peaton guardo el bounding box
                #print(coefficient)
                # bounding_box = (
                #     x * coefficient,
                #     y * coefficient,
                #     (x + win_w) * coefficient,
                #     (y + win_h) * coefficient
                # )
                bounding_box = (
                    x,
                    y,
                    (x + settings.win_w),
                    (y + settings.win_h)
                )
                final_bounding_boxes.append(bounding_box)
                print(bounding_box)
                # bounding_boxes.append(bounding_box)
                final_bounding_boxes.append(bounding_box)
                # count_bounding_boxes += 1

                # Si es un peaton grafico la ventana deslizante
                # if VISUALIZE_SLIDDING_WINDOW:
                #     cv2.rectangle(clone, (x, y), (x + win_w, y + win_h), (0, 255, 0), 2)  # Rectangulo verde
            # Si se quiere ver la ventana deslizante la grafico y la muestro
            # if VISUALIZE_SLIDDING_WINDOW:
            #     cv2.rectangle(clone, (x, y), (x + win_w, y + win_h), (0, 0, 0), 2)  # Rectangulo negro
            #     cv2.imshow("Slidding Window", clone)
            #     cv2.waitKey(1)
            #     time.sleep(0.025)

        # non_result = utils.non_max_suppression_fast(bounding_boxes, 0.3)
        # # print(type(non_result))
        # if non_result.any():
        #     print(non_result)
        #     print(non_result.shape)
        # final_bounding_boxes = np.vstack((final_bounding_boxes, bounding_boxes))
        # final_bounding_boxes += bounding_boxes
        # print("Cantidad de bounding boxes antes de NMS --> {}".format(len(bounding_boxes)))

        # Loop sobre las ventanas deslizantes suprimidas
        # for (startX, startY, endX, endY) in bounding_boxes_suppressed:
        #     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde




        # Si hay bounding boxes productos del NMS graficados, muestro la imagen
        # if len(bounding_boxes_suppressed):
        #     cv2.imshow("Window final", image)
        #     cv2.waitKey(1)
        #     time.sleep(5.025)

        #break    
    # print(final_bounding_boxes)
    # print(final_bounding_boxes.shape)
    
    final_bounding_boxes = utils.non_max_suppression_fast(final_bounding_boxes, 0.25)
    for (startX, startY, endX, endY) in final_bounding_boxes:
        try:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde
            # cv2.rectangle(image, (startY, startX), (endY, endX), (0, 255, 0), 2)  # Rectangulo verde
        except:
            print(sys.exc_info()[0])
            print(startX)
            print(startY)
            print(endX)
            print(endY)
    end = time.time()
    final = end - start
    print("Cantidad de bounding boxes despues de NMS --> {}".format(len(final_bounding_boxes)))

    cv2.imshow("Window final", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Window final")

    print(count)
    print(total)
    print("Promedio: ")
    print(total/count)
    print("Time Finished ...")
    print(final)


if __name__ == '__main__':
    main()