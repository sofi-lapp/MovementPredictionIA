import pygame
import time
import math
import sys
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

# Add the parent directory to path to import steering_system
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from steering_system import SteeringWheelModel
from utils import scale_image, blit_rotate_center, blit_text_center

# Initialize pygame
pygame.init()
pygame.font.init()

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

GRASS = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load(SCRIPT_DIR / "imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/green-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game - Hand Control!")

MAIN_FONT = pygame.font.SysFont("comicsans", 44)

FPS = 60
PATH = [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


class HandController:
    """Controlador de manos para el juego"""
    
    def __init__(self, model_path=None, stats_path=None):
        """Inicializa el controlador de manos"""
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Modelo (si existe)
        self.model = None
        if model_path and stats_path and Path(model_path).exists() and Path(stats_path).exists():
            try:
                self.model = SteeringWheelModel()
                self.model.load_model(model_path, stats_path)
                print("✓ Modelo cargado exitosamente")
            except Exception as e:
                print(f"⚠ No se pudo cargar el modelo: {e}")
                print("→ Usando detección básica de manos")
        
        # Cámara
        self.cam = cv2.VideoCapture(0)
        
        # Suavizado de predicciones
        self.prediction_history = []
        self.history_size = 5
        
        # Estado
        self.current_steering = 0.0  # -1.0 a 1.0
        self.hands_detected = False
    
    def extract_hand_landmarks(self, frame):
        """Extrae landmarks de ambas manos"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        landmarks = np.zeros(126)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx >= 2:
                    break
                    
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                
                start_idx = idx * 63
                landmarks[start_idx:start_idx + 63] = hand_data
        
        return landmarks, results
    
    def smooth_prediction(self, new_prediction):
        """Suaviza las predicciones usando promedio móvil"""
        self.prediction_history.append(new_prediction)
        
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        return np.mean(self.prediction_history)
    
    def update(self):
        """Actualiza el estado del controlador"""
        ret, frame = self.cam.read()
        if not ret:
            return 0.0
        
        # Efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Extraer landmarks
        landmarks, results = self.extract_hand_landmarks(frame)
        
        # Actualizar estado de detección
        self.hands_detected = results.multi_hand_landmarks is not None
        
        # Calcular dirección
        if not self.hands_detected:
            self.current_steering *= 0.9
            steering = self.current_steering
        else:
            if self.model:
                angle = self.model.predict(landmarks)
                steering = self.smooth_prediction(angle)
            else:
                x_positions = []
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    x_positions.append(wrist.x)
                
                avg_x = np.mean(x_positions)
                steering = self.smooth_prediction((avg_x - 0.5) * 2)
            
            self.current_steering = steering
        
        # Mostrar vista de cámara
        self.show_camera_view(frame, steering, results)
        
        return steering
    
    def show_camera_view(self, frame, steering, results):
        """Muestra la vista de la cámara con indicadores"""
        h, w = frame.shape[:2]
        
        # Dibujar landmarks de las manos
        if results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Barra de ángulo
        bar_margin = 50
        bar_y = h - 80
        bar_x_start = bar_margin
        bar_x_end = w - bar_margin
        bar_width = bar_x_end - bar_x_start
        
        # Fondo de la barra
        cv2.rectangle(frame, (bar_x_start, bar_y - 20), 
                      (bar_x_end, bar_y + 20), (50, 50, 50), -1)
        
        # Centro
        center_x = bar_x_start + bar_width // 2
        cv2.line(frame, (center_x, bar_y - 25), 
                 (center_x, bar_y + 25), (255, 255, 255), 2)
        
        # Indicador de posición actual
        current_x = int(center_x + (steering * bar_width / 2))
        
        # Color según ángulo
        if abs(steering) < 0.3:
            color = (0, 255, 0)
        elif abs(steering) < 0.7:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        
        cv2.circle(frame, (current_x, bar_y), 15, color, -1)
        
        # Texto
        angle_text = f"Angulo: {steering:.2f}"
        cv2.putText(frame, angle_text, (bar_x_start, bar_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Etiquetas
        cv2.putText(frame, "-1.0 (IZQ)", (bar_x_start - 30, bar_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "0.0", (center_x - 15, bar_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "+1.0 (DER)", (bar_x_end - 70, bar_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Estado
        status = "MANOS DETECTADAS" if self.hands_detected else "SIN MANOS"
        status_color = (0, 255, 0) if self.hands_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        mode = "Modo: IA" if self.model else "Modo: Basico"
        cv2.putText(frame, mode, (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        cv2.imshow("CONTROL DE MANOS - JUEGO", frame)
        cv2.waitKey(1)
    
    def release(self):
        """Libera recursos"""
        self.cam.release()
        cv2.destroyAllWindows()


class GameInfo:
    LEVELS = 10

    def __init__(self, level=1):
        self.level = level
        self.started = False
        self.level_start_time = 0

    def next_level(self):
        self.level += 1
        self.started = False

    def reset(self):
        self.level = 1
        self.started = False
        self.level_start_time = 0

    def game_finished(self):
        return self.level > self.LEVELS

    def start_level(self):
        self.started = True
        self.level_start_time = time.time()

    def get_level_time(self):
        if not self.started:
            return 0
        return round(time.time() - self.level_start_time)


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def rotate_by_value(self, steering_value):
        """Rota el auto basado en un valor de dirección continuo"""
        rotation = -steering_value * self.rotation_vel
        self.angle += rotation

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def slow_down_collision(self):
        """Reduce velocidad por colisión pero permite seguir moviéndose"""
        self.vel = max(self.vel * 0.5, 0.5)  # Reduce a la mitad, mínimo 0.5


class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (150, 200)

    def __init__(self, max_vel, rotation_vel, path=[]):
        super().__init__(max_vel, rotation_vel)
        self.path = path
        self.current_point = 0
        self.vel = max_vel

    def draw_points(self, win):
        for point in self.path:
            pygame.draw.circle(win, (255, 0, 0), point, 5)

    def draw(self, win):
        super().draw(win)

    def calculate_angle(self):
        target_x, target_y = self.path[self.current_point]
        x_diff = target_x - self.x
        y_diff = target_y - self.y

        if y_diff == 0:
            desired_radian_angle = math.pi / 2
        else:
            desired_radian_angle = math.atan(x_diff / y_diff)

        if target_y > self.y:
            desired_radian_angle += math.pi

        difference_in_angle = self.angle - math.degrees(desired_radian_angle)
        if difference_in_angle >= 180:
            difference_in_angle -= 360

        if difference_in_angle > 0:
            self.angle -= min(self.rotation_vel, abs(difference_in_angle))
        else:
            self.angle += min(self.rotation_vel, abs(difference_in_angle))

    def update_path_point(self):
        target = self.path[self.current_point]
        rect = pygame.Rect(
            self.x, self.y, self.img.get_width(), self.img.get_height())
        if rect.collidepoint(*target):
            self.current_point += 1

    def move(self):
        if self.current_point >= len(self.path):
            return

        self.calculate_angle()
        self.update_path_point()
        super().move()

    def next_level(self, level):
        self.reset()
        self.vel = self.max_vel + (level - 1) * 0.2
        self.current_point = 0


def draw(win, images, player_car, computer_car, game_info, hand_controller):
    for img, pos in images:
        win.blit(img, pos)

    level_text = MAIN_FONT.render(
        f"Level {game_info.level}", 1, (255, 255, 255))
    win.blit(level_text, (10, HEIGHT - level_text.get_height() - 100))

    time_text = MAIN_FONT.render(
        f"Time: {game_info.get_level_time()}s", 1, (255, 255, 255))
    win.blit(time_text, (10, HEIGHT - time_text.get_height() - 70))

    vel_text = MAIN_FONT.render(
        f"Vel: {round(player_car.vel, 1)}px/s", 1, (255, 255, 255))
    win.blit(vel_text, (10, HEIGHT - vel_text.get_height() - 40))

    # Modo de control
    mode_text = "HAND CONTROL" if hand_controller.hands_detected else "KEYBOARD"
    mode_color = (0, 255, 0) if hand_controller.hands_detected else (255, 255, 0)
    mode_surface = MAIN_FONT.render(mode_text, 1, mode_color)
    win.blit(mode_surface, (10, HEIGHT - mode_surface.get_height() - 10))

    player_car.draw(win)
    computer_car.draw(win)
    pygame.display.update()


def move_player(player_car, hand_controller):
    """Mueve el jugador con control de manos o teclado"""
    keys = pygame.key.get_pressed()
    moved = False

    # Control por teclado (siempre disponible)
    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward()
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backward()

    # Control por manos (si hay manos detectadas)
    if hand_controller.hands_detected:
        steering_value = hand_controller.update()
        player_car.rotate_by_value(steering_value)
        moved = True
        player_car.move_forward()
    
    if not moved:
        player_car.reduce_speed()


def handle_collision(player_car, computer_car, game_info):
    if player_car.collide(TRACK_BORDER_MASK) != None:
        player_car.slow_down_collision()  # Reduce velocidad pero sigue moviéndose

    computer_finish_poi_collide = computer_car.collide(
        FINISH_MASK, *FINISH_POSITION)
    if computer_finish_poi_collide != None:
        blit_text_center(WIN, MAIN_FONT, "You lost!")
        pygame.display.update()
        pygame.time.wait(5000)
        game_info.reset()
        player_car.reset()
        computer_car.reset()

    player_finish_poi_collide = player_car.collide(
        FINISH_MASK, *FINISH_POSITION)
    if player_finish_poi_collide != None:
        if player_finish_poi_collide[1] == 0:
            player_car.slow_down_collision()  # Reduce velocidad en colisión
        else:
            game_info.next_level()
            player_car.reset()
            computer_car.next_level(game_info.level)


def main():
    # Buscar modelo entrenado
    model_path = Path(__file__).parent.parent / "models" / "steering_model.h5"
    stats_path = Path(__file__).parent.parent / "models" / "steering_stats.pkl"
    
    print("="*70)
    print("JUEGO DE CARRERAS COMPLETO - CONTROL POR MANOS")
    print("="*70)
    print("\n✨ Características:")
    print("  • Colisiones con bordes")
    print("  • 10 Niveles progresivos")
    print("  • Competencia con IA")
    print("  • Control por manos + teclado")
    print("\nControles:")
    print("  🖐️  Manos: Dirección automática")
    print("  W: Acelerar | S: Retroceder")
    print("  A/D: Girar (teclado alternativo)")
    print("  Q: Salir")
    print("\n" + "="*70)
    
    if model_path.exists() and stats_path.exists():
        print(f"✓ Modelo encontrado")
    else:
        print("⚠ No se encontró modelo entrenado")
        print("→ Usando detección básica")
    print("="*70 + "\n")
    
    # Inicializar
    hand_controller = HandController(
        str(model_path) if (model_path.exists() and stats_path.exists()) else None,
        str(stats_path) if (model_path.exists() and stats_path.exists()) else None
    )
    
    run = True
    clock = pygame.time.Clock()
    images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
              (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
    player_car = PlayerCar(4, 4)
    computer_car = ComputerCar(2, 4, PATH)
    game_info = GameInfo()

    try:
        while run:
            clock.tick(FPS)

            draw(WIN, images, player_car, computer_car, game_info, hand_controller)

            while not game_info.started:
                blit_text_center(
                    WIN, MAIN_FONT, f"Press any key to start level {game_info.level}!")
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        break

                    if event.type == pygame.KEYDOWN:
                        game_info.start_level()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    run = False
                    break

            # Actualizar manos si no están en la fase de inicio
            if game_info.started and not hand_controller.hands_detected:
                hand_controller.update()

            move_player(player_car, hand_controller)
            computer_car.move()

            handle_collision(player_car, computer_car, game_info)

            if game_info.game_finished():
                blit_text_center(WIN, MAIN_FONT, "You won the game!")
                pygame.display.update()
                pygame.time.wait(5000)
                game_info.reset()
                player_car.reset()
                computer_car.reset()

    finally:
        hand_controller.release()
        pygame.quit()
        print("\n¡Hasta luego!")


if __name__ == "__main__":
    main()
