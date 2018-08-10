import utils
import cv2
import time


def main():
    svm = utils.load_checkpoint('/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl')
    (win_w, win_h) = (200, 400)
    predict_funcion = lambda x: svm.predict([x])[0] == 1

    # Abro el video
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        bouding_boxes = utils.detect_pedestrian(frame, win_w, win_h, 1.5, predict_funcion)

        for (startX, startY, endX, endY) in bouding_boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde

        cv2.imshow('frame', frame)  # Muestro el frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
