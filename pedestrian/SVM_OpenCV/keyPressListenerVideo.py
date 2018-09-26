"""
Script para edicion de los parametros en runtime.
Al presionar la tecla 'q' cierra la app QT y CV2
"""
from pynput import keyboard
from threading import Thread
import cv2

"""Funciones para evitar código repetido"""


class ParamsModifier:
    def __init__(self, qt_app, cv2_cap, game):
        self.selected = None
        self.qt_app = qt_app
        self.cv2_cap = cv2_cap
        self.game = game

    def on_press(self, key):
        return

    def on_release(self, key):
        if str(key) == "'q'":
            # Deja de escuchar y cierra la app y CV2
            print('Keypress eliminado')
            self.cv2_cap.release()
            cv2.destroyAllWindows()
            self.qt_app.quit()
            return False
        if str(key) == "'g'":
            self.game.change_active()
            #self.is_game_active = not self.is_game_active

    def wait_for_keypress(self):
        """Bindea el evento de presion de la tecla"""
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
        ) as listener:
            listener.join()


class MyThread(Thread):
    def __init__(self, qt_app, cv2_cap, game):
        Thread.__init__(self)
        self.param_modifier = ParamsModifier(qt_app, cv2_cap, game)

    def run(self):
        self.param_modifier.wait_for_keypress()


def bind_keypress_event(qt_app, cv2_cap, game):
    """Bindea el evento de presion de una tecla"""
    thread = MyThread(qt_app, cv2_cap, game)
    thread.start()
