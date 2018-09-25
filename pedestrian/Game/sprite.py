import numpy as np
import cv2
import numbers




class Sprite:
    """
    # Initialization
    paths=["img/sprites/bee1.png","img/sprites/bee2.png","img/sprites/bee2.png"]
    current_time=round(datetime.utcnow().timestamp() * 1000)
    # generate a sprite with 3 frames, each lasting 500/3 seconds, speed = 20 pixels/second
    sprite=sprite.Sprite.fromPaths(paths,500,current_time=current_time,speed=20)
    sprite.move([100,100]) # set in position (100,100)
    sprite_drawer=sprite.SpriteDrawer()
    ...
    # Sprite update and
    sprite.look_towards_bbox(target) # set direction of sprite towards the center of a bbox
    current_time=round(datetime.utcnow().timestamp() * 1000)
    sprite.update(current_time) # update sprite (anim state and position)
    sprite_drawer.draw(image,sprite)
    """
    @classmethod
    def fromPaths(cls,image_paths, total_duration,current_time,speed=10,loop=True, img_size=(60, 60)):

        images= [cv2.resize(cv2.imread(path,cv2.IMREAD_UNCHANGED), img_size) for path in image_paths]
        return Sprite(images,current_time,total_duration,speed,loop)


    def __init__(self,images,current_time,duration_ms=1000,speed=10,loop=True):
        """

        :param images: list of images, read with opencv
        :param current_time: current time in milliseconds
        :param duration_ms: duration of each image in the animation (or total duration of sprite if each frame's
        duration is the same)
        :param speed: speed in pixels/seconds
        :param loop: True: animation loops through frame. False: animation finishes with last frame
        """
        if isinstance(duration_ms, numbers.Integral):
            n=len(images)
            duration_ms=np.ones(n)*(duration_ms/n)

        self.images=images
        self.duration_ms=duration_ms
        self.position=np.array([0.0,0])
        self.loop=loop
        self.reset_animation(current_time)
        self.speed=speed
        self.direction=np.zeros(2)

    def reset_animation(self,current_time):
        self.frame_elapsed = 0
        self.last_update = current_time
        self.current_frame = 0

    def move(self,position):
        self.position[0]=position[0]
        self.position[1] = position[1]

    def set_speed(self,speed):
        self.speed=speed

    def look_towards(self,direction):
        self.direction[0] = direction[0]
        self.direction[1] = direction[1]
        norm = np.linalg.norm(self.direction)
        if norm > 1e-12:
            self.direction /= norm

    def look_towards_bbox(self,bbox):
        (top, right, bottom, left) = bbox
        center = np.array([(left+right) / 2, (top+bottom) / 2])
        self.look_towards_position(center)

    def look_towards_position(self,position):
        direction=position.astype("float")-self.position
        self.look_towards(direction)

    def in_last_frame(self):
        return self.current_frame==len(self.images)-1

    def update(self,time_ms):
        elapsed=time_ms-self.last_update
        self.last_update = time_ms
        self.update_animation(elapsed)
        self.update_position(elapsed)

    def update_position(self,elapsed):
        self.position+=self.direction*(self.speed*elapsed/1000)

    def update_animation(self,elapsed):
        self.frame_elapsed+=elapsed

        while self.frame_elapsed-self.duration_ms[self.current_frame]>0:
            self.frame_elapsed -= self.duration_ms[self.current_frame]
            if not self.loop and self.in_last_frame():
                break
            if self.loop and self.in_last_frame():
                self.current_frame=0
            else:
                self.current_frame += 1



    def get_current_image(self):
        return self.images[self.current_frame]

    def get_position(self):
        return self.position

    def is_point_in_sprite(self, point):
        curr_image_shape = self.get_current_image().shape[0:2]
        return  point[0] >= self.position[0] and point[0] <= self.position[0] + curr_image_shape[1] and point[1] >= self.position[1] and point[1] <= self.position[1] + curr_image_shape[0]

class SpriteDrawer:
    def __init__(self):
        pass

    def draw(self,image,sprite):
        sprite_image=sprite.get_current_image()
        position=sprite.get_position()
        offset=np.array(sprite_image.shape[0:2])/2

        drawing_position=position-offset
        self.overlay_image_alpha(image,sprite_image,drawing_position)

    def overlay_image_alpha(self,img, img_overlay, pos):
        """Overlay img_overlay on top of img at the position specified by
        pos.

        Alpha mask must contain values within the range [0, 1] and be the
        same size as img_overlay.
        """
        assert(img_overlay.shape[2]==4)
        assert(img.shape[2] == 3)

        x, y = pos.astype("int")

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels = img.shape[2]
        channels=min(channels,3)
        alpha = 0.5#alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            sub_sprite=img_overlay[y1o:y2o, x1o:x2o, c]
            mask=img_overlay[y1o:y2o, x1o:x2o, 3]>0
            sub_image=img[y1:y2, x1:x2, c]
            img[y1:y2, x1:x2, c] = sub_image * (1-mask) + sub_sprite*mask
