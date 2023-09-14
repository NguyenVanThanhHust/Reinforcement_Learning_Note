import os
import time
import cv2
from queue import Queue

LEFT = -1
RIGHT = 1
UP = 2
DOWN = -2

MOVES = {
    LEFT: (1, 0), 
    RIGHT: (-1, 0), 
    UP: (0, -1), 
    DOWN: (0, 1)
}

KEY_TO_DIRECTION = {
    ord("a"): LEFT,
    ord("w"): UP,
    ord("s"): DOWN, 
    ord("d"): RIGHT    
}

class Snake:
    def __init__(self, start_position=[40, 40], length=3):
        self.length = length
        assert len(self.length) <= 5, "max size "
        self.body = [start_position, (start_position[0] + 1, start_position[1]), (start_position[0] + 2, start_position[1])]
        self.extend_body = False
        self.food_queue = []
        self.direction = LEFT
        
    def run_one_step(self, new_direction=None):
        if new_direction is not None:
            if abs(new_direction) != abs(self.direction):
                self.direction = new_direction
        new_head = self.body[0] + MOVES[self.direction]
        self.body.insert(0, new_head)
        if not self.extend_body:
            self.body.pop(-1)
        else:
            self.extend_body = False
        if not self.food_queue.empty():
            oldest_food = self.food_queue[0]
            if oldest_food == self.body[-1]:
                self.extend_body = True
                self.food_queue.pop(0)
    
    def feed_food(self, position=None):
        self.food_queue.append(position)
    
    def check_self_eat(self):
        if self.body[0] in self.body[1:]:
            return True
        return False
    
class SnakeGame:
    def __init__(self, height, width, time_step=1):
        self.width = width
        self.height = height
        self.time_step = time_step
        self.snake = Snake(start_position=[int(width/2), int(height/2)])
        self.score = 0
        self.food_on_screen = False
        self.new_direction = None
        self.food_position = None 
        
    def place_food(self):
        return 

    def show_start(self):
        return 
        
    def show_end(self):
        return 
    
    def show_current_step(self):
        return 
        
    def play(self, ):
        self.show_start()
        time.sleep(self.time_step)
        if not self.food_on_screen:
            self.place_food()
        key = cv2.waitKey(1)
        if key is not None:
            self.new_direction = KEY_TO_DIRECTION(key)
            self.snake.feed_food(self.food_position)            
        return 