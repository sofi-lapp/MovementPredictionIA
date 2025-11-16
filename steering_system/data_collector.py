"""
Módulo para recopilación de datos de entrenamiento del volante virtual
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
from .config import (
    DATA_DIR, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, 
    MIN_TRACKING_CONFIDENCE
)
from .ui_utils import draw_steering_interface


class SteeringWheelDataCollector:
    """Recopila datos de posiciones de manos y ángulos de volante"""
    
    def __init__(self, data_dir=DATA_DIR):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Datos recopilados
        self.hand_positions = []
        self.steering_angles = []
        
        # Estado del volante virtual
        self.current_angle = 0.0  # entre -1 y 1

    def extract_hand_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        landmarks = np.zeros(126)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx >= 2:
                    break
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.extend([lm.x, lm.y, lm.z])
                start = idx * 63
                landmarks[start:start+63] = hand_data

        return landmarks, results
    
    def collect_training_data(self, num_samples=200):
        """
        Secuencia de ida y vuelta:
        Derecha(1.0) → Centro(0) → Izquierda(-1.0) → Centro(0) → Derecha(1.0) ...
        """
        cam = cv2.VideoCapture(0)
        print("\nMODO ENTRENAMIENTO: Derecha → Centro → Izquierda → Centro\n")

        # Crear patrón de ida y vuelta
        # Calcular cuántas muestras por segmento
        samples_per_segment = num_samples // 4
        
        segment1 = np.linspace(1.0, 0.0, samples_per_segment)      # Derecha → Centro
        segment2 = np.linspace(0.0, -1.0, samples_per_segment)     # Centro → Izquierda
        segment3 = np.linspace(-1.0, 0.0, samples_per_segment)     # Izquierda → Centro
        segment4 = np.linspace(0.0, 1.0, samples_per_segment)      # Centro → Derecha
        
        # Concatenar todos los segmentos
        target_angles = np.concatenate([segment1, segment2, segment3, segment4])
        
        # Ajustar para tener exactamente num_samples
        if len(target_angles) < num_samples:
            # Agregar ángulos adicionales repitiendo el patrón
            remaining = num_samples - len(target_angles)
            extra = np.linspace(1.0, 0.0, remaining)
            target_angles = np.concatenate([target_angles, extra])
        else:
            target_angles = target_angles[:num_samples]

        # Animación más rápida
        anim_steps = 8
        anim_wait_ms = 15

        from .ui_utils import draw_steering_wheel_visual

        # Inicializar en el primer ángulo (Derecha = 1.0)
        self.current_angle = target_angles[0]

        for i, target in enumerate(target_angles):

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Salida por teclado.")
                break

            # --------------------------
            # 1. ANIMACIÓN SUAVE
            # --------------------------
            start = self.current_angle
            delta = target - start

            for s in range(1, anim_steps + 1):
                self.current_angle = start + delta * (s / anim_steps)

                ret, frame = cam.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Dibujar landmarks
                _, results = self.extract_hand_landmarks(frame)
                if results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hl, self.mp_hands.HAND_CONNECTIONS
                        )

                # UI
                frame = draw_steering_interface(frame, self.current_angle)
                frame = draw_steering_wheel_visual(frame, self.current_angle)

                # Mostrar contador de muestras
                cv2.putText(frame, f"Muestra: {i+1}/{num_samples}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Entrenamiento", frame)

                if cv2.waitKey(anim_wait_ms) & 0xFF == ord('q'):
                    cam.release()
                    cv2.destroyAllWindows()
                    return None, None

            # --------------------------
            # 2. PARAR (ya está detenido)
            # --------------------------

            # --------------------------
            # 3. CAPTURAR UNA ÚNICA FOTO
            # --------------------------
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            landmarks, results = self.extract_hand_landmarks(frame)

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hl, self.mp_hands.HAND_CONNECTIONS
                    )

            # Guardar
            self.hand_positions.append(landmarks)
            self.steering_angles.append(self.current_angle)

            print(f"[{i+1}/{num_samples}] Ángulo {self.current_angle:+.2f}")

            # UI final
            frame = draw_steering_interface(frame, self.current_angle)
            frame = draw_steering_wheel_visual(frame, self.current_angle)
            
            # Mostrar contador de muestras
            cv2.putText(frame, f"Muestra: {i+1}/{num_samples}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Entrenamiento", frame)

            # Espera corta antes del siguiente ángulo (evita capturas rápidas)
            if cv2.waitKey(90) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        # --------------------------
        # GUARDAR ARCHIVOS
        # --------------------------
        if len(self.hand_positions) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            X = np.array(self.hand_positions)
            y = np.array(self.steering_angles)

            np.save(os.path.join(self.data_dir, f"hand_positions_{timestamp}.npy"), X)
            np.save(os.path.join(self.data_dir, f"steering_angles_{timestamp}.npy"), y)

            print("\nDatos guardados.")
            print(f"Muestras: {len(X)}")
            return X, y

        return None, None