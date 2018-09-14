import utils
import cv2
import numpy as np
from datetime import datetime as dt

COUNT_IGNORED_FRAMES = 3
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'

# Parametros de tracking
THRESHOLD_TRACKING = 0.6
BOUNDING_BOX_LIFE = 50000  # En microsegundos

svm = utils.load_checkpoint(CHECKPOINT_PATH)


def predict_funcion(x):
    """Devuelve true si fue evaluado como peaton"""
    return svm.predict([x])[0] == 1


def predict_proba_funcion(x):
    """Devuelve true si fue evaluado como peaton
    utilizando probabilidades"""
    res = svm.decision_function([x])
    return res[0] > 2


def main():
    (win_w, win_h) = (200, 400)
    old_bounding_boxes = np.array([]) # Para le tracking

    # Abro el video
    cap = cv2.VideoCapture(0)
    count_frames = 0

    while cap.isOpened():
        # Ignora la cantidad de frames seteados
        if count_frames < COUNT_IGNORED_FRAMES:
            count_frames += 1
            cap.grab()
            continue

        # Reinicio el contador y obtengo el nuevo frame
        count_frames = 0
        ret, frame = cap.retrieve()

        # Calculo el tiempo de procesamiento de un frame
        begin_time_frame = dt.now()

        # Obtengo los bounding boxes de los peatones del frame
        bounding_boxes = utils.detect_pedestrian(frame, win_w, win_h, 1.5, predict_proba_funcion)

        # Obtengo los bounding boxes con parametros de tracking
        old_bounding_boxes = utils.tracking_bounding_boxes_ms(old_bounding_boxes, bounding_boxes, THRESHOLD_TRACKING, begin_time_frame, BOUNDING_BOX_LIFE)

        # Dibujo los bounding boxes de los peatones
        for (startX, startY, endX, endY, bounding_box_lifetime) in old_bounding_boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde

        # Muestro el frame
        cv2.imshow('frame', frame)

        # Si presiona 'q' sale
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libero los recursos del video
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
