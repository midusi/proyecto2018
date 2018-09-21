"""
Script para edicion de los parametros en runtime.
Al presionar las teclas F1 .. F5, aparecera un prompt.
F1: edita el parametro scaleDetection   # SVM OpenCV/settings.py.ejemplo
F2: edita el parametro resize           # SVM OpenCV/settings.py.ejemplo
F3: edita el parametro padding          # SVM OpenCV/settings.py.ejemplo
F4: edita el parametro winStride        # SVM OpenCV/settings.py.ejemplo
F5: edita los parametros win_h y win_w  # SVM/settings.py.ejemplo
"""

import settings

from pynput import keyboard
from threading import Thread
import tkinter as tk
from tkinter import simpledialog

"""Funciones para evitar código repetido"""


def ask_integer(title, actual, min_value=0, max_value=100):
    application_window = tk.Tk()
    application_window.withdraw()
    answer = simpledialog.askinteger(title, "Actual: {}; Min: {}; Max: {}".format(actual, min_value, max_value),
                                     parent=application_window, minvalue=min_value, maxvalue=max_value)
    application_window.destroy()
    return answer


def ask_float(title, actual, min_value=0.0, max_value=1.0):
    application_window = tk.Tk()
    application_window.withdraw()
    answer = simpledialog.askfloat(title, "Actual: {}; Min: {}; Max: {}".format(actual, min_value, max_value),
                                   parent=application_window, minvalue=min_value, maxvalue=max_value)
    application_window.destroy()
    return answer


"""FIN - Funciones para evitar código repetido"""


def edit_scale_detection():
    answer = ask_float("Editar scaleDetection", settings.scaleDetection, 0.0, 50.0)
    if answer is not None:
        settings.scaleDetection = answer


def edit_resize():
    answer = ask_float("Editar resize", settings.resize, 0.0, 5.0)
    if answer is not None:
        settings.resize = answer


def edit_padding():
    answer = ask_integer("Editar padding", settings.padding, 0, 500)
    if answer is not None:
        settings.padding = answer


def edit_win_stride():
    answer = ask_integer("Editar winStride", settings.winStride, 0, 500)
    if answer is not None:
        settings.winStride = answer


def edit_window_size():
    new_width = ask_integer("Editar window width", settings.winWidth, 0, 10000)
    new_height = ask_integer("Editar window height", settings.winHeight, 0, 10000)
    if new_width is not None:
        settings.winWidth = new_width
    if new_height is not None:
        settings.winHeight = new_height


def on_press(key):
    if key == keyboard.Key.f12:
        return False
    elif key == keyboard.Key.f1:
        edit_scale_detection()
    elif key == keyboard.Key.f2:
        edit_resize()
    elif key == keyboard.Key.f3:
        edit_padding()
    elif key == keyboard.Key.f4:
        edit_win_stride()
    elif key == keyboard.Key.f5:
        edit_window_size()


def on_release(key):
    if str(key) == "'q'":
        # Deja de escuchar
        print('Keypress eliminado')
        return False


def wait_for_keypress():
    """Bindea el evento de presion de la tecla"""
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release
    ) as listener:
        listener.join()


class MyThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        wait_for_keypress()


def bind_keypress_event():
    """Bindea el evento de presion de una tecla"""
    thread = MyThread()
    thread.start()
