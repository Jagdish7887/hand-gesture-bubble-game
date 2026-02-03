import cv2
import mediapipe as mp
import random
import time
import numpy as np
import math

# --- Configuration ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
STARTING_SPEED = 5
MAX_SPEED = 15
SPAWN_RATE = 25  # Frames between spawns

# Colors (B, G, R)
COLOR_PLAYER = (0, 255, 0)     # Green
COLOR_BOMB   = (0, 0, 255)     # Red
COLOR_GOLD   = (0, 215, 255)   # Gold
COLOR_TEXT   = (255, 255, 255)

class Particle:
    """Creates a small explosion effect."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.life = 1.0  # Life starts at 100%
        self.decay = random.uniform(0.05, 0.1)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay

    def draw(self, img):
        if self.life > 0:
            alpha = int(self.life * 255)
            # Draw faded circle (simulated by shrinking size)
            size = int(5 * self.life)
            cv2.circle(img, (int(self.x), int(self.y)), size, self.color, -1)

class GameObject:
    """Represents Fruits, Bombs, or Gold."""
    def __init__(self, obj_type):
        self.x = random.randint(50, WINDOW_WIDTH - 50)
        self.y = -50
        self.type = obj_type # 'normal', 'bomb', 'gold'
        
        if self.type == 'bomb':
            self.color = COLOR_BOMB
            self.radius = 35
        elif self.type == 'gold':
            self.color = COLOR_GOLD
            self.radius = 25
        else:
            self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            self.radius = 30

class ARGamePro:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, WINDOW_WIDTH)
        self.cap.set(4, WINDOW_HEIGHT)
        
        # Game Variables
        self.reset_game()
        
        # Smoothing variables
        self.prev_x, self.prev_y = 0, 0

    def reset_game(self):
        self.score = 0
        self.lives = 3
        self.game_active = False
        self.objects = []
        self.particles = []
        self.speed = STARTING_SPEED
        self.game_over_timer = 0

    def spawn_object(self):
        # 10% chance of Bomb, 5% chance of Gold, 85% Normal
        rand = random.random()
        if rand < 0.10:
            self.objects.append(GameObject('bomb'))
        elif rand < 0.15:
            self.objects.append(GameObject('gold'))
        else:
            self.objects.append(GameObject('normal'))

    def update(self, finger_pos):
        # Increase difficulty based on score
        self.speed = min(MAX_SPEED, STARTING_SPEED + (self.score // 10))
        
        # Update Particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        # Update Objects
        for obj in self.objects[:]:
            obj.y += self.speed
            
            # Remove if off screen
            if obj.y > WINDOW_HEIGHT:
                self.objects.remove(obj)
                continue
            
            # Collision Detection
            if finger_pos:
                fx, fy = finger_pos
                dist = math.hypot(fx - obj.x, fy - obj.y)
                
                if dist < (obj.radius + 15):
                    # HIT!
                    self.create_explosion(obj.x, obj.y, obj.color)
                    
                    if obj.type == 'bomb':
                        self.lives -= 1
                        self.objects.remove(obj)
                        if self.lives <= 0:
                            self.end_game()
                    elif obj.type == 'gold':
                        self.score += 5
                        self.objects.remove(obj)
                    else:
                        self.score += 1
                        self.objects.remove(obj)

    def create_explosion(self, x, y, color):
        for _ in range(10): # Spawn 10 particles
            self.particles.append(Particle(x, y, color))

    def end_game(self):
        self.game_active = False
        self.game_over_timer = time.time()

    def run(self):
        frame_count = 0
        
        while True:
            success, img = self.cap.read()
            if not success: break
            
            # Flip and process
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            finger_pos = None

            # --- Hand Tracking & Smoothing ---
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark[8] # Index tip
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Smooth the movement (reduces jitter)
                if self.prev_x == 0: self.prev_x, self.prev_y = cx, cy
                cx = int(self.prev_x * 0.5 + cx * 0.5)
                cy = int(self.prev_y * 0.5 + cy * 0.5)
                self.prev_x, self.prev_y = cx, cy
                
                finger_pos = (cx, cy)
                
                # Draw Finger Cursor
                cv2.circle(img, (cx, cy), 15, COLOR_PLAYER, 2)
                cv2.circle(img, (cx, cy), 5, COLOR_PLAYER, -1)
                # Draw "Laser" line from bottom
                cv2.line(img, (WINDOW_WIDTH//2, WINDOW_HEIGHT), (cx, cy), (0, 255, 0, 100), 1)

            # --- Game Logic ---
            if self.game_active:
                frame_count += 1
                if frame_count % SPAWN_RATE == 0:
                    self.spawn_object()
                
                self.update(finger_pos)
                
                # Draw Elements
                for obj in self.objects:
                    # Draw Bomb (X shape inside circle)
                    if obj.type == 'bomb':
                        cv2.circle(img, (int(obj.x), int(obj.y)), obj.radius, obj.color, -1)
                        cv2.putText(img, "!", (int(obj.x)-10, int(obj.y)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    else:
                        cv2.circle(img, (int(obj.x), int(obj.y)), obj.radius, obj.color, -1)
                        cv2.circle(img, (int(obj.x), int(obj.y)), obj.radius, (255,255,255), 2)
                
                for p in self.particles:
                    p.draw(img)

                # UI Overlay
                cv2.putText(img, f"Score: {self.score}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT, 3)
                cv2.putText(img, f"Lives: {'<3 '*self.lives}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            else:
                # Start Screen / Game Over Screen
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                
                if self.score == 0 and self.game_over_timer == 0:
                    title = "FINGER SLASH AR"
                    sub = "Press SPACE to Start"
                else:
                    title = "GAME OVER"
                    sub = f"Final Score: {self.score} | Press SPACE"
                
                cv2.putText(img, title, (WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_GOLD, 4)
                cv2.putText(img, sub, (WINDOW_WIDTH//2 - 250, WINDOW_HEIGHT//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)

            cv2.imshow("AR Game Pro", img)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 32: # Space
                if not self.game_active:
                    self.reset_game()
                    self.game_active = True

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = ARGamePro()
    game.run()