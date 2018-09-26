SHOOTER_POS_Y = 400
MIN_TOP = 10
MAX_TOP = 100
DISPLAY_WIDTH = 640
RESET_POSITION = [-150, DISPLAY_WIDTH+150]
MIN_DIST = 10
import numpy as np
import sprite
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

        self.down_ball = sprite.Sprite.fromPaths(
            ["./fireball/down/tile048.png","./fireball/down/tile049.png","./fireball/down/tile050.png","./fireball/down/tile051.png","./fireball/down/tile052.png","./fireball/down/tile053.png","./fireball/down/tile054.png","./fireball/down/tile055.png"],
            300,
            current_time,
            speed=0,
            img_size=(40,40)
        )
        self.target_position = None

        explosion_paths = []
        for i in range(0,64):
            explosion_paths.append("./explosion/tile0"+(("0"+str(i)) if i <10 else str(i))+".png")

        self.explosion = sprite.Sprite.fromPaths(
            explosion_paths,
            400,
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
        self.down_ball.set_speed(70)

    def show_explosion(self,pos=None,time=round(dt.utcnow().timestamp() * 1000)):
        self.down_ball.set_speed(0)
        self.explosion.reset_animation(time)
        self.explosion.move(pos)
        self.draw_fireball = False
        self.showing_explosion = True

    def update(self, time):
        if(self.was_fired and self.draw_fireball):
            self.down_ball.update(time)
            dist = math.hypot(self.target_position[0] - self.down_ball.get_position()[0], self.target_position[1] - self.down_ball.get_position()[1])
            if(dist < MIN_DIST):
                self.show_explosion(pos=self.down_ball.get_position(), time=time)

        if(self.showing_explosion):
            self.explosion.update(time)

    def get_rect(self):
        return (
            self.down_ball.get_position()[0],
            self.down_ball.get_position()[1],
            self.down_ball.get_position()[0] + self.down_ball.get_current_image().shape[0],
            self.down_ball.get_position()[1] + self.down_ball.get_current_image().shape[1])

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
        self.left_sprite = sprite.Sprite.fromPaths(left_paths, 300, current_time, img_size=(144,100), speed=75)
        self.left_sprite.look_towards((-1, 0))
        #va a la derecha
        self.right_sprite = sprite.Sprite.fromPaths(right_paths, 300, current_time, img_size=(144,100), speed=75)
        self.right_sprite.look_towards((1, 0))

        self.reset_sprites()

        self.alive = True
        self.balls=[FireBall(dragon=self)]

    def get_position(self):
        return self.current_sprite.get_position()

    def update(self, time=None):
        if(time == None):
            time = round(dt.utcnow().timestamp() * 1000)

        if(self.alive):
            self.current_sprite.update(time)

        drawing_balls = False
        for b in self.balls:
            b.update(time)
            drawing_balls = b.was_fired or drawing_balls

        if(not self.alive and not drawing_balls):
            self.dragon_manager.remove_dragon(self)

    def draw(self, image):
        if(self.alive):
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
        if(not self.alive):
            return
        for b in self.balls:
            if(not b.was_fired):
                b.fire(position)
                return True
        return False

    def target_explode(self,pos):
        self.dragon_manager.target_explode(pos)

    def is_colliding_with(self, rect):
        if(not self.alive):
            return
        up_left_point = (rect[0], rect[1])
        up_right_point = (rect[2], rect[1])
        if( self.current_sprite.is_point_in_sprite(up_left_point) or self.current_sprite.is_point_in_sprite(up_right_point)):
            self.alive = False
            return True
        return False

    def check_if_hits(self, shooters):
        for b in self.balls:
            rect = b.get_rect()
            for e in shooters:
                if(e.is_colliding_with(rect)):
                    b.show_explosion(pos=e.get_spawn_explosion_position())
                    break

class DragonManager():
    def __init__(self, sprite_drawer, game):
        self.sprite_drawer = sprite_drawer
        self.game = game
        self.dragons = []
        self.targets = []
    def remove_dragon(self,dragon):
        self.dragons.remove(dragon)
    def count_dragons(self):
        return len(self.dragons)
    def update(self, time):
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
                    shooted = True
                    break
            if(shooted):
                break

    def target_explode(self, pos):
        if( pos in self.targets):
            self.targets.remove(pos)

    def check_if_hits(self, shooters):
        for d in self.dragons:
            d.check_if_hits(shooters)

    def get_dragons(self):
        return self.dragons

    def add_dragon(self):
        self.dragons.append(Dragon(
            ["./dragon/blue/tile009.png","./dragon/blue/tile010.png","./dragon/blue/tile011.png"],
            ["./dragon/blue/tile003.png","./dragon/blue/tile004.png","./dragon/blue/tile005.png"],
            random.choice((True, False)),
            self.sprite_drawer,
            self
        ))

class Shooter():
    def __init__(self, shooter_manager, sprite_drawer, paths, position):
        self.sprite_drawer = sprite_drawer
        self.shooter_manager = shooter_manager
        current_time = round(dt.utcnow().timestamp() * 1000)
        self.start_position = position
        self.sprite = sprite.Sprite.fromPaths(paths, 2500, current_time, speed=0, img_size=(65,80))
        self.sprite.move(self.start_position)
        self.sprite.reset_animation(current_time)
        self.time_to_shoot = 4000
        self.last_shoot = 0
        self.lasers = []
        self.alive = True

    def draw(self, image):
        if(self.alive):
            self.sprite_drawer.draw(image, self.sprite)

        for l in self.lasers:
            self.sprite_drawer.draw(image, l)
        return image
    def update(self, time):
        if(self.alive):
            self.sprite.update(time)
            if(time - self.last_shoot >= self.time_to_shoot):
                self.last_shoot = time
                self.shoot(time)

        for l in self.lasers:
            l.update(time)
            if(l.get_position()[1] < -150):
                self.lasers.remove(l)

        if(len(self.lasers) == 0 and not self.alive):
            self.shooter_manager.remove(self)

    def shoot(self, time):
        laser = sprite.Sprite.fromPaths(["./laser/beams.png"], 1000, time, speed=125, img_size=(13,59))
        laser.look_towards((0,-1))
        laser.move((self.sprite.get_position()[0]+18,self.sprite.get_position()[1]-12))
        self.lasers.append(laser)

    def check_if_hits(self, enemys):
        for l in self.lasers:
            for e in enemys:
                e.is_colliding_with((l.position[0], l.position[1], l.position[0] + l.get_current_image().shape[0], l.position[1]+ l.get_current_image().shape[1]))

    def is_colliding_with(self, rect):
        if(not self.alive):
            return
        down_left_point = (rect[0], rect[3])
        down_right_point = (rect[2], rect[3])
        if( self.sprite.is_point_in_sprite(down_left_point) or self.sprite.is_point_in_sprite(down_right_point)):
            self.alive = False
            return True
        return False

    def get_spawn_explosion_position(self):
        return self.sprite.get_position()

    def move(self, pos_x):
        if(self.alive):
            self.sprite.move((pos_x, self.sprite.get_position()[1]))

class ShootersManager():
    def __init__(self, sprite_drawer, game):
        self.game = game
        self.sprite_drawer = sprite_drawer
        self.shooters = []
        self.millenium_falcon_paths = ["./millenium-falcon/000.png","./millenium-falcon/001.png","./millenium-falcon/002.png","./millenium-falcon/003.png"]

    def update(self, time):
        for s in self.shooters:
            s.update(time)

    def draw(self, image):
        for s in self.shooters:
            image = s.draw(image)
        return image

    def add_shooter(self, position):
        s = Shooter(self, self.sprite_drawer, self.millenium_falcon_paths, position)
        self.shooters.append(s)

    def check_if_hits(self, enemys):
        for s in self.shooters:
            s.check_if_hits(enemys)

    def get_shooters(self):
        return self.shooters

    def count_shooters(self):
        return len(self.shooters)

    def move_shooter(self, index, pos):
        self.shooters[index].move(pos)

    def remove(self, shooter):
        self.shooters.remove(shooter)

class Game():
    def __init__(self):
        self.sprite_drawer = sprite.SpriteDrawer()
        self.dragon_manager = DragonManager(self.sprite_drawer, self)
        self.shooters_manager = ShootersManager(self.sprite_drawer, self)
        self.is_game_active = False

    def update(self, positions_list):
        fire_to = []
        for i in range(len(positions_list)):
            pos_x = positions_list[i][0]+int(positions_list[i][2]//2)
            pos = (pos_x, SHOOTER_POS_Y)
            fire_to.append(pos)
            if(i >= self.shooters_manager.count_shooters()):
                self.shooters_manager.add_shooter(pos)
            if(i >= self.dragon_manager.count_dragons()):
                self.dragon_manager.add_dragon()

            self.shooters_manager.move_shooter(i, pos[0])

        self.dragon_manager.fire_to(fire_to)

        time = round(dt.utcnow().timestamp() * 1000)
        self.dragon_manager.update(time)
        self.shooters_manager.update(time)

        self.shooters_manager.check_if_hits(self.dragon_manager.get_dragons())
        self.dragon_manager.check_if_hits(self.shooters_manager.get_shooters())


    def draw(self, image):
        image = self.dragon_manager.draw(image)
        image = self.shooters_manager.draw(image)
        return image

    def fire_to(self, positions):
        self.dragon_manager.fire_to(positions)

    def add_shooters(self, positions):
        for p in positions:
            self.shooters_manager.add_shooter(p)
    
    def change_active(self):
        self.is_game_active = not self.is_game_active
    
    def get_is_game_active(self):
        return self.is_game_active
