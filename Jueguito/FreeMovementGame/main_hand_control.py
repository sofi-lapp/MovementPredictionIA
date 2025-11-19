import pygame
import time
import math
import sys
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

# Add the MovementPredictionIA directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "MovementPredictionIA"))

from steering_system import SteeringWheelModel
from utils import scale_image, blit_rotate_center

# Initialize pygame
pygame.init()
pygame.font.init()

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

GRASS = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/track-border.png"), 0.9)

RED_CAR = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load(SCRIPT_DIR / "imgs/green-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game - Hand Control!")

FPS = 60


class HandController:
    """Controlador de manos para el juego"""
    
    def __init__(self, model_path=None, stats_path=None):
        """
        Inicializa el controlador de manos
        
        Args:
            model_path: Ruta al modelo entrenado
            stats_path: Ruta a las estadísticas del modelo
        """
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
        
        # Extraer landmarks (necesario para dibujar las manos)
        landmarks, results = self.extract_hand_landmarks(frame)
        
        # Actualizar estado de detección
        self.hands_detected = results.multi_hand_landmarks is not None
        
        # Calcular dirección
        if not self.hands_detected:
            # Sin manos detectadas, volver gradualmente a centro
            self.current_steering *= 0.9
            steering = self.current_steering
        else:
            # Si tenemos modelo entrenado, usarlo
            if self.model:
                angle = self.model.predict(landmarks)
                steering = self.smooth_prediction(angle)
            else:
                # Método básico: usar posición horizontal de las manos
                x_positions = []
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]  # Muñeca
                    x_positions.append(wrist.x)
                
                avg_x = np.mean(x_positions)
                # Convertir de [0, 1] a [-1, 1]
                steering = self.smooth_prediction((avg_x - 0.5) * 2)
            
            self.current_steering = steering
        
        # Mostrar vista de cámara con interfaz completa
        self.show_camera_view(frame, steering, results)
        
        return steering
    
    def show_camera_view(self, frame, steering, results):
        """Muestra la vista de la cámara con indicadores - Estilo entrenamiento"""
        h, w = frame.shape[:2]
        
        # Dibujar landmarks de las manos (como en el entrenamiento)
        if results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Usar la misma interfaz del sistema de entrenamiento
        # Barra de ángulo en la parte inferior
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
        
        # Indicador de posición actual (círculo)
        current_x = int(center_x + (steering * bar_width / 2))
        
        # Color según ángulo
        if abs(steering) < 0.3:
            color = (0, 255, 0)  # Verde - centro
        elif abs(steering) < 0.7:
            color = (0, 165, 255)  # Naranja
        else:
            color = (0, 0, 255)  # Rojo - extremo
        
        cv2.circle(frame, (current_x, bar_y), 15, color, -1)
        
        # Texto de ángulo
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
        
        # Estado en la parte superior
        status = "🖐️ MANOS DETECTADAS" if self.hands_detected else "⚠️ SIN MANOS"
        status_color = (0, 255, 0) if self.hands_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # Modo del modelo
        mode = "Modo: IA (Modelo entrenado)" if self.model else "Modo: Basico (Posicion manos)"
        cv2.putText(frame, mode, (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        cv2.imshow("CONTROL DE MANOS - JUEGO", frame)
        cv2.waitKey(1)
    
    def release(self):
        """Libera recursos"""
        self.cam.release()
        cv2.destroyAllWindows()


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
        """
        Rota el auto basado en un valor de dirección continuo
        
        Args:
            steering_value: Valor entre -1.0 (izquierda) y 1.0 (derecha)
        """
        # Convertir el valor de dirección a rotación
        # Multiplicar por rotation_vel para obtener la velocidad de rotación
        rotation = -steering_value * self.rotation_vel
        self.angle += rotation

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)


def draw(win, images, player_car, hand_controller):
    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)
    
    # Mostrar información de control
    font = pygame.font.Font(None, 36)
    mode_text = "HAND CONTROL" if hand_controller.hands_detected else "KEYBOARD (W/A/D)"
    color = (0, 255, 0) if hand_controller.hands_detected else (255, 255, 0)
    text_surface = font.render(mode_text, True, color)
    win.blit(text_surface, (10, 10))
    
    # Mostrar velocidad
    speed_text = f"Speed: {player_car.vel:.1f}"
    speed_surface = font.render(speed_text, True, (255, 255, 255))
    win.blit(speed_surface, (10, 50))
    
    pygame.display.update()


def main():
    # Buscar modelo entrenado
    model_path = Path(__file__).parent.parent.parent / "MovementPredictionIA" / "models" / "steering_model.h5"
    stats_path = Path(__file__).parent.parent.parent / "MovementPredictionIA" / "models" / "steering_stats.pkl"
    
    print("="*70)
    print("JUEGO DE CARRERAS - CONTROL POR MANOS")
    print("="*70)
    print("\nControles:")
    print("  Manos: Dirección automática (si hay modelo entrenado)")
    print("  W: Acelerar")
    print("  A/D: Girar (teclado alternativo)")
    print("  Q: Salir")
    print("\n" + "="*70)
    
    if model_path.exists() and stats_path.exists():
        print(f"✓ Modelo encontrado: {model_path}")
    else:
        print("⚠ No se encontró modelo entrenado")
        print("→ Usando detección básica de posición de manos")
        if not model_path.parent.exists():
            print(f"→ Directorio de modelos no existe: {model_path.parent}")
        else:
            print(f"→ Entrena el modelo primero ejecutando:")
            print(f"   /home/eme/Desktop/IA-Code/.venv/bin/python /home/eme/Desktop/IA-Code/MovementPredictionIA/main.py")
    print("="*70 + "\n")
    
    # Inicializar
    hand_controller = HandController(
        str(model_path) if (model_path.exists() and stats_path.exists()) else None,
        str(stats_path) if (model_path.exists() and stats_path.exists()) else None
    )
    
    run = True
    clock = pygame.time.Clock()
    images = [(GRASS, (0, 0)), (TRACK, (0, 0))]
    player_car = PlayerCar(4, 4)
    
    try:
        while run:
            clock.tick(FPS)
            
            # Actualizar controlador de manos
            steering_value = hand_controller.update()
            
            draw(WIN, images, player_car, hand_controller)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
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
            if keys[pygame.K_q]:
                run = False
                break
            
            # Control por manos (si hay manos detectadas)
            if hand_controller.hands_detected:
                # Aplicar dirección de las manos
                player_car.rotate_by_value(steering_value)
                # Auto-acelerar cuando se detectan manos
                moved = True
                player_car.move_forward()
            
            if not moved:
                player_car.reduce_speed()
    
    finally:
        hand_controller.release()
        pygame.quit()
        print("\n¡Hasta luego!")


if __name__ == "__main__":
    main()
