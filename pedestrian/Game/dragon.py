MIN_TOP = 10
MAX_TOP = 100
DISPLAY_WIDTH = 640
RESET_POSITION = [-150, DISPLAY_WIDTH+150]
MIN_DIST = 10
import numpy as np
from sprite import *
import random, math
from datetime import datetime as dt

class FireBall():
    def __init__(self, dragon):
        self.dragon = dragon
        self.sprite_drawer = dragon.sprite_drawer
        # self.start_right = dragon.start_right

        self.was_fired = False
        self.draw_fireball = False

        current_time = round(dt.utcnow().timestamp() * 1000)

        self.down_ball = Sprite.fromPaths(
            ["./fireball/down/tile048.png","./fireball/down/tile049.png","./fireball/down/tile050.png","./fireball/down/tile051.png","./fireball/down/tile052.png","./fireball/down/tile053.png","./fireball/down/tile054.png","./fireball/down/tile055.png"],
            500,
            current_time,
            speed=0,
            img_size=(70,70)
        )
        self.target_position = None

        explosion_paths = []
        for i in range(0,64):
            explosion_paths.append("./explosion/tile0"+(("0"+str(i)) if i <10 else str(i))+".png")

        self.explosion = Sprite.fromPaths(
            explosion_paths,
            1500,
            current_time,
            speed=0,
            loop=False,
            img_size=(80,80)
        )
        self.showing_explosion = False


    def fire(self, position):
        self.draw_fireball = self.was_fired = True
        self.target_position = position
        self.down_ball.move(self.dragon.get_position())
        self.down_ball.look_towards_position(np.array(self.target_position))
        self.down_ball.set_speed(40)

    def show_explosion(self,time):
        self.down_ball.set_speed(0)
        self.explosion.reset_animation(time-10)
        self.explosion.move(self.target_position)
        self.draw_fireball = False
        self.showing_explosion = True

    def update(self, time):
        if(self.was_fired and self.draw_fireball):
            self.down_ball.update(time)
            dist = math.hypot(self.target_position[0] - self.down_ball.get_position()[0], self.target_position[1] - self.down_ball.get_position()[1])
            if(dist < MIN_DIST):
                self.show_explosion(time)

        if(self.showing_explosion):
            self.explosion.update(time)

    def draw(self, image):
        if(self.draw_fireball):
            self.sprite_drawer.draw(image, self.down_ball)

        if(self.showing_explosion):
            self.sprite_drawer.draw(image, self.explosion)
            if(self.explosion.in_last_frame()):
                self.showing_explosion = self.draw_fireball = self.was_fired = False
                self.dragon.target_explode(self.target_position)

class Dragon():
    def __init__(self,left_paths, right_paths, start_right, sprite_drawer, dragon_manager=None):
        self.dragon_manager = dragon_manager
        self.start_right = start_right
        self.sprite_drawer = sprite_drawer

        current_time = round(dt.utcnow().timestamp() * 1000)
        #va a la izquierda
        self.left_sprite = Sprite.fromPaths(left_paths, 300, current_time, speed=60)
        self.left_sprite.look_towards((-1, 0))
        #va a la derecha
        self.right_sprite = Sprite.fromPaths(right_paths, 300, current_time, speed=60)
        self.right_sprite.look_towards((1, 0))

        self.reset_sprites()

        self.balls=[FireBall(dragon=self)]

    def get_position(self):
        return self.current_sprite.get_position()

    def update(self, time=None):
        if(time == None):
            time = round(dt.utcnow().timestamp() * 1000)

        self.current_sprite.update(time)

        for b in self.balls:
            b.update(time)

    def draw(self, image):
        self.sprite_drawer.draw(image, self.current_sprite)

        for b in self.balls:
            b.draw(image)

        pos_x = self.current_sprite.get_position()[0]
        if( pos_x <= RESET_POSITION[0] or pos_x >= RESET_POSITION[1]):
            self.start_right = not self.start_right
            self.reset_sprites()

        return image

    def reset_sprites(self):
        self.pos_y = random.randint(MIN_TOP, MAX_TOP)
        self.left_sprite.move(( random.randint(DISPLAY_WIDTH, RESET_POSITION[1]-1), self.pos_y))
        self.right_sprite.move((random.randint(RESET_POSITION[0]+1,0), self.pos_y))
        self.current_sprite = self.right_sprite if self.start_right else self.left_sprite

    def fire_to(self, position):
        for b in self.balls:
            if(not b.was_fired):
                b.fire(position)
                return True
        return False

    def target_explode(self,pos):
        self.dragon_manager.target_explode(pos)

class DragonManager():
    def __init__(self):
        self.sprite_drawer = SpriteDrawer()
        self.dragons = [
            Dragon(
                ["./dragon/blue/tile009.png","./dragon/blue/tile010.png","./dragon/blue/tile011.png"],
                ["./dragon/blue/tile003.png","./dragon/blue/tile004.png","./dragon/blue/tile005.png"],
                True,
                self.sprite_drawer,
                self
            ),
            Dragon(
                ["./dragon/blue/tile009.png","./dragon/blue/tile010.png","./dragon/blue/tile011.png"],
                ["./dragon/blue/tile003.png","./dragon/blue/tile004.png","./dragon/blue/tile005.png"],
                False,
                self.sprite_drawer,
                self
            )
        ]
        self.targets = []
    def update(self):
        time = round(dt.utcnow().timestamp() * 1000)
        for d in self.dragons:
            d.update(time)

    def draw(self, image):
        for i in self.dragons:
            image = i.draw(image)
        return image

    def fire_to(self, positions):
        for pos in positions:
            if( pos in self.targets):
                continue
            self.targets.append(pos)
            shooted = False
            for d in self.dragons:
                if(d.fire_to(pos)):
                    print("fire_to - d")
                    shooted = True
                    break
            if(shooted):
                break

    def target_explode(self, pos):
        if( pos in self.targets):
            self.targets.remove(pos)
