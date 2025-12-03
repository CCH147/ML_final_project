import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math  # <--- 新增這一行

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

# 1. 用於預測下一步座標的輔助函式 (不移動蛇)
    def get_point_in_direction(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # 直走
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # 右轉
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # 左轉

        x = self.head.x
        y = self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        return Point(x, y)

    # 2. 洪水填充算法 (Flood Fill) - 計算可活動空間
    def get_accessible_area(self, point):
        # 如果該點本身就是撞牆或撞身體，空間為 0
        if self.is_collision(point):
            return 0

        # 準備 BFS
        # 將像素座標轉換為網格索引 (Grid Index)
        start_x = int(point.x // BLOCK_SIZE)
        start_y = int(point.y // BLOCK_SIZE)
        
        cols = self.w // BLOCK_SIZE
        rows = self.h // BLOCK_SIZE
        
        queue = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        
        # 建立障礙物集合 (蛇的身體)
        obstacles = set()
        for pt in self.snake:
            ox = int(pt.x // BLOCK_SIZE)
            oy = int(pt.y // BLOCK_SIZE)
            obstacles.add((ox, oy))
            
        count = 0
        # 算出「是否大於蛇身長度」
        limit = len(self.snake) + 5 

        while queue:
            cx, cy = queue.pop(0)
            count += 1
            
            # 如果空間已經足夠大，足以容納整條蛇，就不用再算了，視為安全
            if count > limit:
                return limit

            neighbors = [
                (cx+1, cy), (cx-1, cy),
                (cx, cy+1), (cx, cy-1)
            ]
            
            for nx, ny in neighbors:
                # 檢查邊界
                if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
                    continue
                # 檢查障礙與訪問
                if (nx, ny) in visited or (nx, ny) in obstacles:
                    continue
                
                visited.add((nx, ny))
                queue.append((nx, ny))
                
        return count

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        dist_before = math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            dist_after = math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
            
            # 生存成本 (步數懲罰)：每走一步都扣一點分，逼它走最短路徑
            step_penalty = -0.05

            if dist_after < dist_before:
            # 靠近食物
                reward = 1 # 1
            else:
                reward = -1.5 # 1.5
            
            # 將步數懲罰加進去
            reward += step_penalty

        #if not self.is_tail_reachable():

            # 給予一個僅次於死亡 (-10) 的重罰，例如 -5 或 -2
            
        #w    reward = -5.0 
            
            # 可以在這裡直接結束遊戲
            # game_over = True 
            # 但建議先給重罰 讓它有機會在絕境中掙扎

        # ==========================================

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

 # 加入 game.py 的 SnakeGameAI 類別中
    def is_tail_reachable(self):
        # 1. 準備起點 (蛇頭) 與 終點 (蛇尾)
        head_x = int(self.head.x // BLOCK_SIZE)
        head_y = int(self.head.y // BLOCK_SIZE)
        start = (head_x, head_y)
        
        # 蛇尾是列表的最後一個元素
        tail = self.snake[-1]
        tail_x = int(tail.x // BLOCK_SIZE)
        tail_y = int(tail.y // BLOCK_SIZE)
        target = (tail_x, tail_y)
        
        # 2. 設定地圖邊界
        cols = self.w // BLOCK_SIZE
        rows = self.h // BLOCK_SIZE
        
        # 3. BFS 初始化
        queue = [start]
        visited = set()
        visited.add(start)
        
        # 4. 設定障礙物
        obstacles = set()
        # self.snake[:-1] 從頭取到倒數第二個，不包含最後一個(尾巴)
        for pt in self.snake[:-1]: 
            ox = int(pt.x // BLOCK_SIZE)
            oy = int(pt.y // BLOCK_SIZE)
            obstacles.add((ox, oy))
            
        # 5. 開始 BFS 搜尋
        while queue:
            cx, cy = queue.pop(0)
            
            # 如果走到目標 (尾巴)，回傳 True
            if (cx, cy) == target:
                return True
            
            # 檢查上下左右
            neighbors = [
                (cx+1, cy), (cx-1, cy),
                (cx, cy+1), (cx, cy-1)
            ]
            
            for nx, ny in neighbors:
                # 檢查邊界
                if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
                    continue
                # 檢查障礙與訪問紀錄
                if (nx, ny) in visited or (nx, ny) in obstacles:
                    continue
                
                visited.add((nx, ny))
                queue.append((nx, ny))
                
        # 6. 找遍全圖都走不到尾巴 -> 危險
        return False

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        # [在 game.py 中新增這個函式]
    # 這就是你要的：只負責算座標，不負責移動
    def get_next_head_point(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right Turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn

        x = self.head.x
        y = self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        return Point(x, y)
