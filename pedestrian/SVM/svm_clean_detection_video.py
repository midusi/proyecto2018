import utils
import cv2

COUNT_IGNORED_FRAMES = 3
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'
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

    # Abro el video
    cap = cv2.VideoCapture(0)
    count_frames = 0
    while cap.isOpened():
        # Ignora los frames
        if count_frames < COUNT_IGNORED_FRAMES:
            count_frames += 1
            cap.grab()
            continue

        count_frames = 0  # Reinicio el contador
        ret, frame = cap.retrieve()

        # Obtengo los bounding boxes de los peatones del frame
        bounding_boxes = utils.detect_pedestrian(frame, win_w, win_h, 1.5, predict_proba_funcion)

        # Dibujo los peatones
        for (startX, startY, endX, endY) in bounding_boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde

        # Muestro el frame
        cv2.imshow('frame', frame)

        # Si presiona 'q' sale
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
