import utils
import cv2
import time
import numpy as np


VISUALIZE_SLIDDING_WINDOW = True


def main():
    """La mayoria del codigo se saco de
    https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/"""
    # Tamaño de ventana deslizante
    (win_w, win_h) = (200, 400)

    # Cargo y preparo la imagen
    image = utils.load_image_from_path('/home/genaro/PycharmProjects/proyecto2018/pedestrian/SVM/imgs/street4.jpg')
    image = utils.to_grayscale(image)
    image = utils.normalize_image_max(image)

    # Obtengo mi SVM
    svm = utils.load_checkpoint('/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl')

    # Loop sobre las diferentes escalas
    for resized_image in utils.get_pyramid(image, scale=1.5):
        clone = resized_image.copy()  # Hago una copia de la imagen original para graficar detecciones
        clone_suppression = clone.copy()  # Hago una copia para graficar el resultado de NMS

        # Lista de bounding boxes para el Non Maximal Suppression
        bounding_boxes = []

        # Llevo control para saber si funciona el NMS
        count_bounding_boxes = 0

        # Loop sobre la ventana deslizante en diferentes posiciones
        for (x, y, window) in utils.get_sliding_window(resized_image, stepSize=(32, 64), windowSize=(win_w, win_h)):
            # Si la ventana no coincide con nuestro tamaño de ventana, se ignora
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != win_h or window.shape[1] != win_w:
                continue

            # Inteligencia ACA!
            # Manejo la ventana deslizante actual
            cropped_image = utils.crop_image(image, x, y, win_w, win_h)  # Recorto la ventana
            cropped_image = utils.resize(cropped_image, [96, 48])  # Escalo
            cropped_image_hog = utils.get_hog_from_image(cropped_image, normalize=False)  # Obtengo el HOG

            # Comienzo a predecir
            prediction = svm.predict([cropped_image_hog])[0]
            if prediction == 1:

                # Si es un peaton guardo el bounding box
                bounding_box = (x, y, x + win_w, y + win_h)
                bounding_boxes.append(bounding_box)
                count_bounding_boxes += 1

                # Si es un peaton grafico la ventana deslizante
                if VISUALIZE_SLIDDING_WINDOW:
                    cv2.rectangle(clone, (x, y), (x + win_w, y + win_h), (0, 255, 0), 2)  # Rectangulo verde

            if VISUALIZE_SLIDDING_WINDOW:
                cv2.rectangle(clone, (x, y), (x + win_w, y + win_h), (0, 0, 0), 2)  # Rectangulo negro
                cv2.imshow("Slidding Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)

        bounding_boxes = np.array(bounding_boxes)
        pick = utils.non_max_suppression_fast(bounding_boxes, 0.3)

        print("Cantidad de bounding boxes antes de NMS --> {}".format(len(bounding_boxes)))

        # Loop sobre las ventanas deslizantes suprimidas
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(clone_suppression, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde

        print("Cantidad de bounding boxes despues de NMS --> {}".format(len(pick)))

        # Si hay bounding boxes productos del NMS graficados, muestro la imagen
        if len(pick):
            cv2.imshow("Window final", clone_suppression)
            cv2.waitKey(1)
            time.sleep(5.025)


if __name__ == '__main__':
    main()