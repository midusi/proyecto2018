"""Valores para testeo del script"""
porcentaje = .55
resize = 0.3
scaleDetection = 1.5
padding = 16
winStride = 5
window_width = 200
window_height = 200


from pynput import keyboard

import tkinter as tk
from tkinter import simpledialog
"""Funciones para evitar código repetido"""
def ask_integer(title, actual, min_value=0, max_value=100):
    application_window = tk.Tk()
    application_window.withdraw()
    answer = simpledialog.askinteger(title, "Actual: {}; Min: {}; Max: {}".format(actual, min_value, max_value), parent=application_window, minvalue=min_value, maxvalue=max_value)
    application_window.destroy()
    return answer

def ask_float(title, actual, min_value=0.0, max_value=1.0):
    application_window = tk.Tk()
    application_window.withdraw()
    answer = simpledialog.askfloat(title, "Actual: {}; Min: {}; Max: {}".format(actual, min_value, max_value), parent=application_window, minvalue=min_value, maxvalue=max_value)
    application_window.destroy()
    return answer
"""FIN - Funciones para evitar código repetido"""


def edit_scale_detection():
    global scaleDetection
    answer = ask_float("Editar scaleDetection", scaleDetection, 0.0, 50.0)
    if(answer is not None):
        scaleDetection = answer
        print("Scale detection actualizado a {}".format(scaleDetection))

def edit_resize():
    global resize
    answer = ask_float("Editar resize", resize, 0.0, 5.0)
    if(answer is not None):
        resize = answer
        print("Resize actualizado a {}".format(resize))

def edit_padding():
    global padding
    answer = ask_integer("Editar padding", padding, 0, 500)
    if(answer is not None):
        padding = answer
        print("padding actualizado a {}".format(padding))

def edit_win_stride():
    global winStride
    answer = ask_integer("Editar winStride", winStride, 0, 500)
    if(answer is not None):
        winStride = answer
        print("winStride actualizado a {}".format(winStride))

def edit_window_size():
    global window_width
    global window_height
    new_width   = ask_integer("Editar window width", window_width, 0, 10000)
    new_height  = ask_integer("Editar window height", window_height, 0, 10000)
    if(new_width is not None):
        window_width = new_width
    if(new_height is not None):
        window_height = new_height

def on_press(key):
    if(key == keyboard.Key.f12):
        return False
    elif(key == keyboard.Key.f1):
        edit_scale_detection()
    elif(key == keyboard.Key.f2):
        edit_resize()
    elif(key == keyboard.Key.f3):
        edit_padding()
    elif(key == keyboard.Key.f4):
        edit_win_stride()
    elif(key == keyboard.Key.f5):
        edit_window_size()

with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()
