from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
import sys
import cv2

from qtconsole.qt import QtGui


class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("Reconocimiento de peatones - Informática UNLP ")

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.showMaximized()

        # video_widget = QVideoWidget()
        #
        # # Create a widget for window contents
        # wid = QWidget(self)
        # self.setCentralWidget(wid)
        #
        # # Create layouts to place inside widget
        # # controlLayout = QHBoxLayout()
        # # controlLayout.setContentsMargins(0, 0, 0, 0)
        # # controlLayout.addWidget(self.playButton)
        # # controlLayout.addWidget(self.positionSlider)
        #
        # layout = QVBoxLayout()
        # layout.addWidget(video_widget)
        # # layout.addLayout(controlLayout)
        # # layout.addWidget(self.errorLabel)
        #
        # # Set widget to contain window contents
        # wid.setLayout(layout)
        #
        # cap = cv2.VideoCapture(0)
        # while cap.isOpened():
        #     cap.grab()
        #
        #     ret, frame = cap.retrieve()
        #
        #     videoFrame = QtGui.QImage(frame, 200, 200,
        #                               QtGui.QImage.Format_RGB888)
        #     convertFrame = QtGui.QPixmap(videoFrame)
        #     self.imageBox.setPixmap(convertFrame)
        #     self.imageBox.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())
