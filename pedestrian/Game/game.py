from sprite import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
import utils
import cv2
import numpy as np
from datetime import datetime as dt
from dragon import *

# Parametros del SVM
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'

# Parametros de video
COUNT_IGNORED_FRAMES = 3
# SAVE_POSITIVE_PATH = '/home/genaro/PycharmProjects/predicciones_positivas/'
SAVE_POSITIVE_PATH = None

# Parametros de tracking
THRESHOLD_TRACKING = 0.6
BOUNDING_BOX_LIFE = 50000  # En microsegundos

# Parametros de deteccion
SCORE_THRESHOLD = 5  # Threshold del score de la deteccion

# Instancio el SVM de forma global para facil acceso
#svm = utils.load_checkpoint(CHECKPOINT_PATH)


def predict_funcion(x):
    """Devuelve true si fue evaluado como peaton"""
    return svm.predict([x])[0] == 1


def predict_proba_funcion(x):
    """Devuelve true si fue evaluado como peaton
    utilizando probabilidades"""
    res = svm.decision_function([x])
    return res[0] > SCORE_THRESHOLD


def main():
    dragon_manager = DragonManager()
    # paths = []
    # for i in range (22):
    #     i_s = str(i)
    #     paths.append("bee/tile0" +  ("0"+i_s if i < 10 else i_s)+".png")
    #
    # bee = Sprite.fromPaths(paths, 1500, current_time=round(dt.now().timestamp() * 1000))
    # bee.move((400,400))
    # sd = SpriteDrawer()


    (win_w, win_h) = (200, 400)

    # Abro el video
    cap = cv2.VideoCapture(0)
    count_frames = 0

    while cap.isOpened():
        # Ignora la cantidad de frames seteados
        if count_frames < COUNT_IGNORED_FRAMES:
            count_frames += 1
        else:
            count_frames = 0
            b = [(200.0,200.0),(210.0,190.0)]
            dragon_manager.fire_to(b)
        #     cap.grab()
        #     continue

        cap.grab()
        # Reinicio el contador y obtengo el nuevo frame
        # count_frames = 0
        ret, frame = cap.retrieve()


        # current_time=round(dt.utcnow().timestamp() * 1000)
        # bee.update(current_time)
        # sd.draw(frame, bee)
        dragon_manager.update()
        fram = dragon_manager.draw(frame)
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
