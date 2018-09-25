"""
Script para edicion de los parametros en runtime.
Al presionar las teclas F1 .. F5, aparecera un prompt.
F1: edita el parametro scaleDetection   # SVM OpenCV/settings.py.ejemplo
F2: edita el parametro resize           # SVM OpenCV/settings.py.ejemplo
F3: edita el parametro padding          # SVM OpenCV/settings.py.ejemplo
F4: edita el parametro winStride        # SVM OpenCV/settings.py.ejemplo
F5: edita los parametros winHeight # SVM/settings.py.ejemplo
F6: edita los parametros winWidth  # SVM/settings.py.ejemplo
F7: edita los parametros scoreThreshold  # SVM/settings.py.ejemplo
F8: edita los parametros countIgnoredFrames  # SVM/settings.py.ejemplo
F9: edita los parametros trackThreshold  # SVM/settings.py.ejemplo
F10: edita los parametros boundBoxLife  # SVM/settings.py.ejemplo
"""

import settings

from pynput import keyboard
from threading import Thread

"""Funciones para evitar código repetido"""


class ParamsModifier:
    def __init__(self):
        self.selected = None

    def edit_value(self, increment):
        """Incrementa o decrementa el valor seleccionado"""
        value_to_add = 1 if increment else -1
        updated_value = None
        if self.selected == 'scaleDetection':
            updated_value = settings.scaleDetection = settings.scaleDetection + value_to_add
        elif self.selected == 'resize':
            updated_value = settings.resize = settings.resize + value_to_add
        elif self.selected == 'padding':
            updated_value = settings.padding = settings.padding + value_to_add
        elif self.selected == 'winStride':
            updated_value = settings.winStride = settings.winStride + value_to_add
        elif self.selected == 'winHeight':
            updated_value = settings.winHeight = settings.winHeight + value_to_add
        elif self.selected == 'winWidth':
            updated_value = settings.winWidth = settings.winWidth + value_to_add
        elif self.selected == 'scoreThreshold':
            updated_value = settings.scoreThreshold = settings.scoreThreshold + value_to_add
        elif self.selected == 'countIgnoredFrames':
            updated_value = settings.countIgnoredFrames = settings.countIgnoredFrames + value_to_add
        elif self.selected == 'trackThreshold':
            updated_value = settings.trackThreshold = settings.trackThreshold + value_to_add
        elif self.selected == 'boundBoxLife':
            updated_value = settings.boundBoxLife = settings.boundBoxLife + value_to_add

        if updated_value:
            print('Valor nuevo -> {}'.format(updated_value))

    def on_press(self, key):
        if key == keyboard.Key.f1:
            self.selected = 'scaleDetection'
        elif key == keyboard.Key.f2:
            self.selected = 'resize'
        elif key == keyboard.Key.f3:
            self.selected = 'padding'
        elif key == keyboard.Key.f4:
            self.selected = 'winStride'
        elif key == keyboard.Key.f5:
            self.selected = 'winHeight'
        elif key == keyboard.Key.f6:
            self.selected = 'winWidth'
        elif key == keyboard.Key.f7:
            self.selected = 'scoreThreshold'
        elif key == keyboard.Key.f8:
            self.selected = 'countIgnoredFrames'
        elif key == keyboard.Key.f9:
            self.selected = 'trackThreshold'
        elif key == keyboard.Key.f10:
            self.selected = 'boundBoxLife'
        elif key == keyboard.Key.down:
            self.edit_value(False)
        elif key == keyboard.Key.up:
            self.edit_value(True)

        if self.selected:
            print('Parametro seleccionado -> {}'.format(self.selected))

    @staticmethod
    def on_release(key):
        if str(key) == "'q'":
            # Deja de escuchar
            print('Keypress eliminado')
            return False

    def wait_for_keypress(self):
        """Bindea el evento de presion de la tecla"""
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
        ) as listener:
            listener.join()


class MyThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        param_modifier = ParamsModifier()
        param_modifier.wait_for_keypress()


def bind_keypress_event():
    """Bindea el evento de presion de una tecla"""
    thread = MyThread()
    thread.start()
