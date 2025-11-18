import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
from .config import (
    DATA_DIR, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, 
    MIN_TRACKING_CONFIDENCE, calculate_angle_distribution
)
from .ui_utils import draw_steering_interface, run_f1_countdown

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
        cam = cv2.VideoCapture(0)
        
        #Calcular distribución y número de vueltas
        distribution, num_laps = calculate_angle_distribution(num_samples)
        
        print(f"\nMODO ENTRENAMIENTO:")
        print(f"Muestras totales: {num_samples}")
        print(f"Vueltas completas: {num_laps}")
        print(f"\nDistribución (muestras por ángulo):")
        print(f"  Centro (0.0): {distribution[0.0]} muestras")
        print(f"  Extremos (-1.0, +1.0): {distribution[-1.0]}, {distribution[1.0]} muestras")
        print(f"  Intermedios: 1-{min([v for k, v in distribution.items() if abs(k) > 0.1 and abs(k) < 0.9])} muestras")
        #print(f"\nPatrón: 0 → -1.0 → 0 → +1.0 → 0 (más lento en centro/extremos)\n")
        
        # Construir secuencia: 0 → -1.0 → 0 → +1.0 → 0, repetir
        all_angles_sorted = sorted(distribution.keys())  # De -1.0 a +1.0
        
        target_angles = []
        pause_times = []
        
        for lap in range(num_laps):
            # Fase 1: Centro (0.0) → Izquierda (-1.0)
            angles_center_to_left = [a for a in reversed(all_angles_sorted) if a <= 0 and distribution[a] > lap]
            
            for angle in angles_center_to_left:
                target_angles.append(angle)
                
                # Calcular tiempo de pausa según prioridad del ángulo
                abs_angle = abs(angle)
                if abs_angle < 0.05:  # Centro
                    pause_time = 300  # Pausa larga (ms)
                elif abs_angle > 0.95:  # Extremos
                    pause_time = 250  # Pausa media-larga
                elif abs_angle > 0.75 or abs_angle < 0.25:  # Cerca de centro/extremos
                    pause_time = 150  # Pausa media
                else:  # Intermedios
                    pause_time = 60   # Pausa corta
                
                pause_times.append(pause_time)
            
            # Fase 2: Izquierda (-1.0) → Centro (0.0)
            angles_left_to_center = [a for a in all_angles_sorted if a < 0 and distribution[a] > lap]
            
            for angle in angles_left_to_center:
                target_angles.append(angle)
                
                abs_angle = abs(angle)
                if abs_angle < 0.05:
                    pause_time = 300
                elif abs_angle > 0.95:
                    pause_time = 250
                elif abs_angle > 0.75 or abs_angle < 0.25:
                    pause_time = 150
                else:
                    pause_time = 60
                
                pause_times.append(pause_time)
            
            # Agregar el centro antes de ir a la derecha
            if distribution[0.0] > lap:
                target_angles.append(0.0)
                pause_times.append(300)
            
            # Fase 3: Centro (0.0) → Derecha (+1.0)
            angles_center_to_right = [a for a in all_angles_sorted if a > 0 and distribution[a] > lap]
            
            for angle in angles_center_to_right:
                target_angles.append(angle)
                
                abs_angle = abs(angle)
                if abs_angle < 0.05:
                    pause_time = 300
                elif abs_angle > 0.95:
                    pause_time = 250
                elif abs_angle > 0.75 or abs_angle < 0.25:
                    pause_time = 150
                else:
                    pause_time = 60
                
                pause_times.append(pause_time)
            
            # Fase 4: Derecha (+1.0) → Centro (0.0) para volver al inicio
            angles_right_to_center = [a for a in reversed(all_angles_sorted) if a > 0 and distribution[a] > lap]
            
            for angle in angles_right_to_center:
                target_angles.append(angle)
                
                abs_angle = abs(angle)
                if abs_angle < 0.05:
                    pause_time = 300
                elif abs_angle > 0.95:
                    pause_time = 250
                elif abs_angle > 0.75 or abs_angle < 0.25:
                    pause_time = 150
                else:
                    pause_time = 60
                
                pause_times.append(pause_time)
        
        # Ajustar para tener exactamente num_samples (por si hay redondeo)
        target_angles = target_angles[:num_samples]
        pause_times = pause_times[:num_samples]

        # Animación más rápida
        anim_steps = 8
        anim_wait_ms = 15

        from .ui_utils import draw_steering_wheel_visual

        # SEMAFORO - CUENTA REGRESIVA DE 20 SEGUNDOS
        countdown_completed = run_f1_countdown(cam)
        
        if not countdown_completed:
            print("\nEntrenamiento cancelado antes de comenzar.")
            cam.release()
            cv2.destroyAllWindows()
            return None, None
        
        print("\n¡Capturas iniciada!\n")
        
        # Inicializar en el centro (0.0)
        self.current_angle = 0.0

        for i, target in enumerate(target_angles):

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Salida por teclado.")
                break

            # 1. ANIMACIÓN SUAVE
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

            # 2. PARAR (ya está detenido)

            # 3. CAPTURAR UNA ÚNICA FOTO
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

            # Espera ajustada según el ángulo (más tiempo en centro/extremos)
            if cv2.waitKey(pause_times[i]) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        #GUARDAR ARCHIVOS
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