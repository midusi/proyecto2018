import utils
import cv2

COUNT_IGNORED_FRAMES = 3
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint (Bruce Willis).pkl'


def main():
    svm = utils.load_checkpoint(CHECKPOINT_PATH)
    (win_w, win_h) = (200, 400)
    predict_funcion = lambda x: svm.predict([x])[0] == 1

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
        bouding_boxes = utils.detect_pedestrian(frame, win_w, win_h, 1.5, predict_funcion)

        # Dibujo los peatones
        for (startX, startY, endX, endY) in bouding_boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde

        # Muestro el frame
        cv2.imshow('frame', frame)
        # exit()
        # Si presiona 'q' sale
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
