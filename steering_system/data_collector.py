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
        """
        Inicializa el recopilador de datos
        
        Args:
            data_dir: Directorio donde guardar los datos
        """
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
        self.current_angle = 0.0  # Entre -1 y 1
        
    def extract_hand_landmarks(self, frame):
        """
        Extrae landmarks de ambas manos
        
        Args:
            frame: Frame de la cámara
        
        Returns:
            tuple: (landmarks array, mediapipe results)
                - landmarks: numpy array con 126 valores (2 manos × 21 landmarks × 3 coords)
                - results: Objeto de resultados de MediaPipe
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # Array para almacenar landmarks de ambas manos
        landmarks = np.zeros(126)  # 2 manos × 21 landmarks × 3 coords
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx >= 2:  # Solo procesamos 2 manos máximo
                    break
                    
                # Extraer coordenadas
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                
                # Colocar en la posición correcta del array
                start_idx = idx * 63  # 21 landmarks × 3 coords
                landmarks[start_idx:start_idx + 63] = hand_data
        
        return landmarks, results
    
    def collect_training_data(self, num_samples=200):
        """
        Recopila datos de entrenamiento de forma MANUAL
        
        Args:
            num_samples: Número de muestras objetivo a capturar
        
        Returns:
            tuple: (X, y) Arrays numpy con los datos recopilados
        """
        cam = cv2.VideoCapture(0)
        
        print("\n" + "="*70)
        print("MODO ENTRENAMIENTO - VOLANTE VIRTUAL (CAPTURA MANUAL)")
        print("="*70)
        print(f"\nObjetivo: {num_samples} muestras")
        print("\nFLUJO DE TRABAJO:")
        print("  1. Ajusta el angulo con las flechas <- ->")
        print("  2. Posiciona tus manos en la posicion deseada")
        print("  3. Presiona ESPACIO para CAPTURAR la muestra")
        print("  4. Repite hasta completar las muestras")
        print("\nCONTROLES:")
        print("  <- : Girar volante a la IZQUIERDA (-0.05)")
        print("  -> : Girar volante a la DERECHA (+0.05)")
        print("  ^ : Girar RAPIDO a la IZQUIERDA (-0.20)")
        print("  v : Girar RAPIDO a la DERECHA (+0.20)")
        print("  0 : Resetear a centro (0.0)")
        print("  ESPACIO: * CAPTURAR muestra actual *")
        print("  'q': Terminar y guardar")
        print("\nCONSEJOS:")
        print("  - Captura varias muestras para cada angulo")
        print("  - Varia ligeramente la posicion de las manos")
        print("  - Cubre todo el rango: -1.0 a +1.0")
        print("  - Especialmente importante: -1.0, -0.5, 0.0, +0.5, +1.0")
        print("\n" + "="*70 + "\n")
        
        start_time = time.time()
        last_capture_angle = None
        
        while len(self.hand_positions) < num_samples:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Invertir imagen horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            
            # Extraer landmarks
            landmarks, results = self.extract_hand_landmarks(frame)
            
            # Dibujar manos
            hands_detected = False
            if results.multi_hand_landmarks:
                hands_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Actualizar ángulo con teclado
            key = cv2.waitKey(1) & 0xFF
            
            # Flechas para ajustar ángulo
            if key == 81 or key == 2 or key == ord('a'):  # Flecha izquierda o 'a'
                self.current_angle = max(-1.0, self.current_angle - 0.05)
                print(f"<- Angulo ajustado: {self.current_angle:.2f}")
            elif key == 83 or key == 3 or key == ord('d'):  # Flecha derecha o 'd'
                self.current_angle = min(1.0, self.current_angle + 0.05)
                print(f"-> Angulo ajustado: {self.current_angle:.2f}")
            elif key == 82 or key == 0 or key == ord('w'):  # Flecha arriba o 'w' - Giro rápido izquierda
                self.current_angle = max(-1.0, self.current_angle - 0.20)
                print(f"<-<- Angulo ajustado (rapido): {self.current_angle:.2f}")
            elif key == 84 or key == 1 or key == ord('s'):  # Flecha abajo o 's' - Giro rápido derecha
                self.current_angle = min(1.0, self.current_angle + 0.20)
                print(f"->-> Angulo ajustado (rapido): {self.current_angle:.2f}")
            elif key == ord('0') or key == ord('c'):  # Resetear a centro
                self.current_angle = 0.0
                print(f"Angulo reseteado: {self.current_angle:.2f}")
            
            # ESPACIO para capturar muestra
            elif key == ord(' '):
                if hands_detected:
                    self.hand_positions.append(landmarks)
                    self.steering_angles.append(self.current_angle)
                    last_capture_angle = self.current_angle
                    
                    progress = len(self.hand_positions)
                    print(f"OK Muestra {progress}/{num_samples} CAPTURADA | Angulo: {self.current_angle:+.2f}")
                    
                    # Sugerencia de próximos ángulos
                    if progress % 10 == 0:
                        print(f"\n--- {progress} muestras capturadas ---")
                        print("Asegurate de cubrir todos los angulos importantes:")
                        print("  -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, +0.2, +0.4, +0.6, +0.8, +1.0")
                        print()
                else:
                    print("! No se detectan manos! Posiciona tus manos antes de capturar")
            
            elif key == ord('q'):
                print("\n! Saliendo antes de completar todas las muestras...")
                break
            
            # Dibujar interfaz
            frame = draw_steering_interface(frame, self.current_angle)
            
            # Dibujar volante visual rotado
            from .ui_utils import draw_steering_wheel_visual
            frame = draw_steering_wheel_visual(frame, self.current_angle)
            
            # Información en pantalla
            progress_pct = (len(self.hand_positions) / num_samples) * 100
            color = (0, 255, 0) if hands_detected else (0, 0, 255)
            
            cv2.putText(frame, f"Muestras: {len(self.hand_positions)}/{num_samples} ({progress_pct:.0f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Indicador de manos detectadas (comentado - no mostrar)
            # hand_status = "MANOS: SI" if hands_detected else "MANOS: NO"
            # cv2.putText(frame, hand_status, (10, 70), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Última captura
            if last_capture_angle is not None:
                cv2.putText(frame, f"Ultima captura: {last_capture_angle:+.2f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Instrucción principal (comentado - no mostrar)
            # cv2.putText(frame, "Presiona ESPACIO para CAPTURAR", (10, 150), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Entrenamiento - Volante Virtual", frame)
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Guardar datos
        if len(self.hand_positions) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convertir a numpy arrays
            X = np.array(self.hand_positions)
            y = np.array(self.steering_angles)
            
            # Guardar
            np.save(os.path.join(self.data_dir, f"hand_positions_{timestamp}.npy"), X)
            np.save(os.path.join(self.data_dir, f"steering_angles_{timestamp}.npy"), y)
            
            print(f"\nOK Datos guardados:")
            print(f"  - Muestras totales: {len(X)}")
            print(f"  - Rango de angulos: [{y.min():.2f}, {y.max():.2f}]")
            print(f"  - Archivo: steering_data/.._{timestamp}.npy")
            
            # Estadísticas
            left_samples = np.sum(y < -0.1)
            center_samples = np.sum(np.abs(y) <= 0.1)
            right_samples = np.sum(y > 0.1)
            
            print(f"\n  Distribucion:")
            print(f"    Izquierda: {left_samples} ({left_samples/len(y)*100:.1f}%)")
            print(f"    Centro: {center_samples} ({center_samples/len(y)*100:.1f}%)")
            print(f"    Derecha: {right_samples} ({right_samples/len(y)*100:.1f}%)")
        else:
            print("\n! No se recopilaron datos")
        
        return X, y if len(self.hand_positions) > 0 else (None, None)
