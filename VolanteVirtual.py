"""
Sistema de Volante Virtual con Deep Learning
============================================

Este sistema permite controlar un volante virtual usando las manos detectadas por MediaPipe.
Devuelve valores entre -1 (izquierda máxima) y +1 (derecha máxima).

Funcionamiento:
1. MODO ENTRENAMIENTO: Recopila datos de posiciones de manos y el ángulo del volante deseado
2. MODO PREDICCIÓN: Analiza la posición de las manos y predice el ángulo del volante

Características del modelo:
- Input: Landmarks de ambas manos (21 landmarks × 3 coordenadas × 2 manos = 126 valores)
- Output: Valor continuo entre -1 y +1 (regresión)
- Arquitectura: Red neuronal densa con capas dropout
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pickle
import time


class SteeringWheelDataCollector:
    """Recopila datos de posiciones de manos y ángulos de volante"""
    
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.data_dir = "steering_data"
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
        
        Returns:
            numpy array con 126 valores (2 manos × 21 landmarks × 3 coords)
            Si falta una mano, se rellena con ceros
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
    
    def draw_steering_interface(self, frame, angle):
        """Dibuja interfaz de volante virtual en el frame"""
        h, w = frame.shape[:2]
        
        # Dibujar barra de ángulo
        bar_y = h - 80
        bar_x_start = 50
        bar_x_end = w - 50
        bar_width = bar_x_end - bar_x_start
        
        # Fondo de la barra
        cv2.rectangle(frame, (bar_x_start, bar_y - 20), (bar_x_end, bar_y + 20), (50, 50, 50), -1)
        
        # Centro
        center_x = bar_x_start + bar_width // 2
        cv2.line(frame, (center_x, bar_y - 25), (center_x, bar_y + 25), (255, 255, 255), 2)
        
        # Indicador de posición actual
        current_x = int(center_x + (angle * bar_width / 2))
        color = (0, 255, 0) if abs(angle) < 0.3 else (0, 165, 255) if abs(angle) < 0.7 else (0, 0, 255)
        cv2.circle(frame, (current_x, bar_y), 15, color, -1)
        
        # Texto de ángulo
        angle_text = f"Angulo: {angle:.2f}"
        cv2.putText(frame, angle_text, (bar_x_start, bar_y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Etiquetas
        cv2.putText(frame, "-1.0 (IZQ)", (bar_x_start - 30, bar_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "+1.0 (DER)", (bar_x_end - 70, bar_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def collect_training_data(self, num_samples=200):
        """
        Recopila datos de entrenamiento de forma MANUAL
        
        Args:
            num_samples: Número de muestras objetivo a capturar
        """
        cam = cv2.VideoCapture(0)
        
        print("\n" + "="*70)
        print("MODO ENTRENAMIENTO - VOLANTE VIRTUAL (CAPTURA MANUAL)")
        print("="*70)
        print(f"\nObjetivo: {num_samples} muestras")
        print("\nFLUJO DE TRABAJO:")
        print("  1. Ajusta el ángulo con las flechas ← →")
        print("  2. Posiciona tus manos en la posición deseada")
        print("  3. Presiona ESPACIO para CAPTURAR la muestra")
        print("  4. Repite hasta completar las muestras")
        print("\nCONTROLES:")
        print("  ← : Girar volante a la IZQUIERDA (-0.05)")
        print("  → : Girar volante a la DERECHA (+0.05)")
        print("  ↑ : Girar RÁPIDO a la IZQUIERDA (-0.20)")
        print("  ↓ : Girar RÁPIDO a la DERECHA (+0.20)")
        print("  0 : Resetear a centro (0.0)")
        print("  ESPACIO: ★ CAPTURAR muestra actual ★")
        print("  'q': Terminar y guardar")
        print("\nCONSEJOS:")
        print("  - Captura varias muestras para cada ángulo")
        print("  - Varía ligeramente la posición de las manos")
        print("  - Cubre todo el rango: -1.0 a +1.0")
        print("  - Especialmente importante: -1.0, -0.5, 0.0, +0.5, +1.0")
        print("\n" + "="*70 + "\n")
        
        start_time = time.time()
        last_capture_angle = None
        
        while len(self.hand_positions) < num_samples:
            ret, frame = cam.read()
            if not ret:
                break
            
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
                print(f"← Ángulo ajustado: {self.current_angle:.2f}")
            elif key == 83 or key == 3 or key == ord('d'):  # Flecha derecha o 'd'
                self.current_angle = min(1.0, self.current_angle + 0.05)
                print(f"→ Ángulo ajustado: {self.current_angle:.2f}")
            elif key == 82 or key == 0 or key == ord('w'):  # Flecha arriba o 'w' - Giro rápido izquierda
                self.current_angle = max(-1.0, self.current_angle - 0.20)
                print(f"←← Ángulo ajustado (rápido): {self.current_angle:.2f}")
            elif key == 84 or key == 1 or key == ord('s'):  # Flecha abajo o 's' - Giro rápido derecha
                self.current_angle = min(1.0, self.current_angle + 0.20)
                print(f"→→ Ángulo ajustado (rápido): {self.current_angle:.2f}")
            elif key == ord('0') or key == ord('c'):  # Resetear a centro
                self.current_angle = 0.0
                print(f"↺ Ángulo reseteado: {self.current_angle:.2f}")
            
            # ESPACIO para capturar muestra
            elif key == ord(' '):
                if hands_detected:
                    self.hand_positions.append(landmarks)
                    self.steering_angles.append(self.current_angle)
                    last_capture_angle = self.current_angle
                    
                    progress = len(self.hand_positions)
                    print(f"✓ Muestra {progress}/{num_samples} CAPTURADA | Ángulo: {self.current_angle:+.2f}")
                    
                    # Sugerencia de próximos ángulos
                    if progress % 10 == 0:
                        print(f"\n--- {progress} muestras capturadas ---")
                        print("Asegúrate de cubrir todos los ángulos importantes:")
                        print("  -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, +0.2, +0.4, +0.6, +0.8, +1.0")
                        print()
                else:
                    print("⚠ No se detectan manos! Posiciona tus manos antes de capturar")
            
            elif key == ord('q'):
                print("\n⚠ Saliendo antes de completar todas las muestras...")
                break
            
            # Dibujar interfaz
            frame = self.draw_steering_interface(frame, self.current_angle)
            
            # Información en pantalla
            progress_pct = (len(self.hand_positions) / num_samples) * 100
            color = (0, 255, 0) if hands_detected else (0, 0, 255)
            
            cv2.putText(frame, f"Muestras: {len(self.hand_positions)}/{num_samples} ({progress_pct:.0f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Indicador de manos detectadas
            hand_status = "MANOS: SI ✓" if hands_detected else "MANOS: NO ✗"
            cv2.putText(frame, hand_status, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Última captura
            if last_capture_angle is not None:
                cv2.putText(frame, f"Ultima captura: {last_capture_angle:+.2f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Instrucción principal
            cv2.putText(frame, "Presiona ESPACIO para CAPTURAR", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
            
            print(f"\n✓ Datos guardados:")
            print(f"  - Muestras totales: {len(X)}")
            print(f"  - Rango de ángulos: [{y.min():.2f}, {y.max():.2f}]")
            print(f"  - Archivo: steering_data/.._{timestamp}.npy")
            
            # Estadísticas
            left_samples = np.sum(y < -0.1)
            center_samples = np.sum(np.abs(y) <= 0.1)
            right_samples = np.sum(y > 0.1)
            
            print(f"\n  Distribución:")
            print(f"    Izquierda: {left_samples} ({left_samples/len(y)*100:.1f}%)")
            print(f"    Centro: {center_samples} ({center_samples/len(y)*100:.1f}%)")
            print(f"    Derecha: {right_samples} ({right_samples/len(y)*100:.1f}%)")
        else:
            print("\n⚠ No se recopilaron datos")
        
        return X, y


class SteeringWheelModel:
    """Modelo de regresión para predecir ángulo del volante"""
    
    def __init__(self, input_size=126):
        """
        Args:
            input_size: Tamaño del input (126 para 2 manos con 21 landmarks cada una)
        """
        self.input_size = input_size
        self.model = None
        self.history = None
        
        # Estadísticas para normalización
        self.mean = None
        self.std = None
    
    def build_model(self):
        """
        Construye red neuronal para regresión
        
        Arquitectura optimizada para regresión continua:
        - Capas densas con activación ReLU
        - BatchNormalization para estabilidad
        - Dropout para regularización
        - Salida lineal para valores continuos
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.input_size,)),
            
            # Normalización
            layers.BatchNormalization(),
            
            # Primera capa densa - extracción de características
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Segunda capa - combinación de características
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Tercera capa - refinamiento
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Cuarta capa
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Salida - regresión lineal
            layers.Dense(1, activation='tanh')  # tanh para rango [-1, 1]
        ])
        
        # Compilar con MSE para regresión
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        self.model = model
        return model
    
    def load_all_training_data(self, data_dir="steering_data"):
        """Carga todos los datos de entrenamiento disponibles"""
        X_list = []
        y_list = []
        
        # Buscar archivos de datos
        position_files = [f for f in os.listdir(data_dir) if f.startswith('hand_positions_')]
        
        if not position_files:
            raise ValueError("No se encontraron datos de entrenamiento. Ejecuta primero el modo de recopilación.")
        
        for pos_file in position_files:
            timestamp = pos_file.replace('hand_positions_', '').replace('.npy', '')
            angle_file = f'steering_angles_{timestamp}.npy'
            
            # Cargar datos
            X = np.load(os.path.join(data_dir, pos_file))
            y = np.load(os.path.join(data_dir, angle_file))
            
            X_list.append(X)
            y_list.append(y)
            
            print(f"✓ Cargado: {pos_file} ({len(X)} muestras)")
        
        # Concatenar todos los datos
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        
        print(f"\nTotal de muestras: {len(X_all)}")
        
        return X_all, y_all
    
    def normalize_data(self, X, fit=False):
        """Normaliza los datos de entrada"""
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-8  # Evitar división por cero
        
        return (X - self.mean) / self.std
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Entrena el modelo
        
        Args:
            X: Datos de entrada (posiciones de manos)
            y: Valores objetivo (ángulos del volante)
            validation_split: Porcentaje para validación
            epochs: Número de épocas
            batch_size: Tamaño del batch
        """
        # Normalizar datos
        X_normalized = self.normalize_data(X, fit=True)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X_normalized, y, test_size=validation_split, random_state=42
        )
        
        print(f"\n{'='*60}")
        print(f"Datos de entrenamiento: {X_train.shape}")
        print(f"Datos de validación: {X_val.shape}")
        print(f"{'='*60}\n")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_steering_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        print("Iniciando entrenamiento...\n")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar
        val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS FINALES:")
        print(f"  MSE (Loss): {val_loss:.4f}")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  Precisión estimada: ±{val_mae:.2f} en rango [-1, 1]")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_model(self, model_path='steering_model.h5', stats_path='steering_stats.pkl'):
        """Guarda el modelo y estadísticas de normalización"""
        self.model.save(model_path)
        
        with open(stats_path, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'std': self.std
            }, f)
        
        print(f"✓ Modelo guardado en {model_path}")
        print(f"✓ Estadísticas guardadas en {stats_path}")
    
    def load_model(self, model_path='steering_model.h5', stats_path='steering_stats.pkl'):
        """Carga el modelo y estadísticas"""
        self.model = keras.models.load_model(model_path)
        
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
        
        print(f"✓ Modelo cargado desde {model_path}")
    
    def predict(self, hand_landmarks):
        """
        Predice el ángulo del volante
        
        Args:
            hand_landmarks: Array de landmarks (126 valores)
        
        Returns:
            angle: Valor entre -1 y +1
        """
        # Normalizar
        landmarks_normalized = (hand_landmarks - self.mean) / self.std
        
        # Predecir
        landmarks_batch = np.expand_dims(landmarks_normalized, axis=0)
        prediction = self.model.predict(landmarks_batch, verbose=0)
        
        # Asegurar que esté en el rango [-1, 1]
        angle = np.clip(prediction[0][0], -1.0, 1.0)
        
        return angle


class RealTimeSteeringPredictor:
    """Predictor en tiempo real para el volante virtual"""
    
    def __init__(self, model):
        """
        Args:
            model: Modelo entrenado (SteeringWheelModel)
        """
        self.model = model
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Suavizado de predicciones
        self.prediction_history = []
        self.history_size = 5
    
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
    
    def draw_steering_interface(self, frame, angle):
        """Dibuja interfaz de volante"""
        h, w = frame.shape[:2]
        
        # Barra de ángulo
        bar_y = h - 80
        bar_x_start = 50
        bar_x_end = w - 50
        bar_width = bar_x_end - bar_x_start
        
        # Fondo
        cv2.rectangle(frame, (bar_x_start, bar_y - 20), (bar_x_end, bar_y + 20), (50, 50, 50), -1)
        
        # Centro
        center_x = bar_x_start + bar_width // 2
        cv2.line(frame, (center_x, bar_y - 25), (center_x, bar_y + 25), (255, 255, 255), 2)
        
        # Indicador
        current_x = int(center_x + (angle * bar_width / 2))
        color = (0, 255, 0) if abs(angle) < 0.3 else (0, 165, 255) if abs(angle) < 0.7 else (0, 0, 255)
        cv2.circle(frame, (current_x, bar_y), 15, color, -1)
        
        # Texto
        cv2.putText(frame, f"VOLANTE: {angle:.3f}", (bar_x_start, bar_y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Etiquetas
        cv2.putText(frame, "-1.0", (bar_x_start - 10, bar_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "0.0", (center_x - 15, bar_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "+1.0", (bar_x_end - 20, bar_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Dirección
        if angle < -0.1:
            direction = "← IZQUIERDA"
            dir_color = (0, 255, 255)
        elif angle > 0.1:
            direction = "DERECHA →"
            dir_color = (255, 255, 0)
        else:
            direction = "↑ RECTO"
            dir_color = (0, 255, 0)
        
        cv2.putText(frame, direction, (w // 2 - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, dir_color, 3)
        
        return frame
    
    def run(self, show_console_output=True):
        """
        Ejecuta predicción en tiempo real
        
        Args:
            show_console_output: Si True, imprime valores en consola
        """
        cam = cv2.VideoCapture(0)
        
        print("\n" + "="*70)
        print("MODO PREDICCIÓN - VOLANTE VIRTUAL")
        print("="*70)
        print("\nControla el volante con tus manos!")
        print("Valores: -1.0 (izquierda) ... 0.0 (centro) ... +1.0 (derecha)")
        print("\nPresiona 'q' para salir\n")
        print("="*70 + "\n")
        
        if show_console_output:
            print("Valores del volante (tiempo real):")
            print("-" * 50)
        
        frame_count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Extraer landmarks
            landmarks, results = self.extract_hand_landmarks(frame)
            
            # Dibujar manos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Predecir ángulo
            angle = self.model.predict(landmarks)
            angle_smoothed = self.smooth_prediction(angle)
            
            # Salida por consola
            if show_console_output and frame_count % 10 == 0:  # Cada 10 frames
                print(f"Volante: {angle_smoothed:+.3f}  |  ", end="")
                
                # Barra visual en consola
                bar_length = 40
                center = bar_length // 2
                pos = int(center + angle_smoothed * center)
                bar = ['-'] * bar_length
                bar[center] = '|'
                bar[pos] = '█'
                print(''.join(bar))
            
            # Dibujar interfaz
            frame = self.draw_steering_interface(frame, angle_smoothed)
            
            # Info adicional
            cv2.putText(frame, "Prediccion en Tiempo Real", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Volante Virtual - Predicción", frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("Predicción finalizada")
        print("="*70)


# ============================================================================
# MENÚ PRINCIPAL
# ============================================================================

def main():
    """Función principal con menú de opciones"""
    
    print("\n" + "="*70)
    print("SISTEMA DE VOLANTE VIRTUAL CON DEEP LEARNING")
    print("="*70)
    print("\n🎮 Controla un volante virtual con tus manos")
    print("📊 Salida: valores entre -1 (izquierda) y +1 (derecha)")
    print("\nOpciones:")
    print("  1. 📹 Recopilar datos de entrenamiento")
    print("  2. 🧠 Entrenar modelo")
    print("  3. 🚗 Predicción en tiempo real")
    print("  4. ⚡ Flujo rápido (recopilar + entrenar + predecir)")
    print("  5. ℹ️  Información del sistema")
    
    option = input("\nSelecciona una opción (1-5): ")
    
    if option == '1':
        # RECOPILACIÓN DE DATOS
        print("\n" + "="*70)
        print("RECOPILACIÓN DE DATOS")
        print("="*70)
        
        num_samples = int(input("\nNúmero de muestras a capturar (recomendado 100-300): "))
        
        collector = SteeringWheelDataCollector()
        collector.collect_training_data(num_samples=num_samples)
        
    elif option == '2':
        # ENTRENAMIENTO
        print("\n" + "="*70)
        print("ENTRENAMIENTO DEL MODELO")
        print("="*70)
        
        model = SteeringWheelModel()
        
        try:
            # Cargar datos
            X, y = model.load_all_training_data()
            
            # Construir modelo
            model.build_model()
            print("\nResumen del modelo:")
            model.model.summary()
            
            # Entrenar
            epochs = int(input("\nNúmero de épocas (recomendado 100): "))
            model.train(X, y, epochs=epochs)
            
            # Guardar
            model.save_model()
            
            print("\n✓ Entrenamiento completado exitosamente")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Asegúrate de haber recopilado datos primero (opción 1)")
    
    elif option == '3':
        # PREDICCIÓN
        print("\n" + "="*70)
        print("PREDICCIÓN EN TIEMPO REAL")
        print("="*70)
        
        model = SteeringWheelModel()
        
        try:
            model.load_model()
            
            show_console = input("\n¿Mostrar valores en consola? (s/n): ").lower() == 's'
            
            predictor = RealTimeSteeringPredictor(model)
            predictor.run(show_console_output=show_console)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Asegúrate de haber entrenado el modelo primero (opción 2)")
    
    elif option == '4':
        # FLUJO COMPLETO
        print("\n" + "="*70)
        print("FLUJO COMPLETO")
        print("="*70)
        print("\nEste proceso incluye:")
        print("  1. Recopilación de datos (manual)")
        print("  2. Entrenamiento del modelo")
        print("  3. Predicción en tiempo real")
        
        input("\nPresiona ENTER para continuar...")
        
        # 1. Recopilar datos
        print("\n[1/3] Recopilando datos...")
        collector = SteeringWheelDataCollector()
        collector.collect_training_data(num_samples=150)
        
        # 2. Entrenar
        print("\n[2/3] Entrenando modelo...")
        model = SteeringWheelModel()
        X, y = model.load_all_training_data()
        model.build_model()
        model.train(X, y, epochs=100)
        model.save_model()
        
        # 3. Predecir
        print("\n[3/3] Iniciando predicción...")
        input("Presiona ENTER para comenzar...")
        predictor = RealTimeSteeringPredictor(model)
        predictor.run(show_console_output=True)
    
    elif option == '5':
        # INFORMACIÓN
        print("\n" + "="*70)
        print("INFORMACIÓN DEL SISTEMA")
        print("="*70)
        print("\n📋 Descripción:")
        print("  Sistema de Deep Learning para simular un volante virtual")
        print("  usando la posición de las manos detectadas por MediaPipe.")
        print("\n🔧 Funcionamiento:")
        print("  - Input: 126 valores (2 manos × 21 landmarks × 3 coords)")
        print("  - Arquitectura: Red neuronal densa con 4 capas ocultas")
        print("  - Output: Valor continuo entre -1 y +1")
        print("    • -1.0 = Volante girado completamente a la izquierda")
        print("    •  0.0 = Volante en posición central (recto)")
        print("    • +1.0 = Volante girado completamente a la derecha")
        print("\n🎮 Captura de Datos (MANUAL):")
        print("  1. Ajusta el ángulo con flechas ← → (o W A S D)")
        print("  2. Posiciona tus manos como si condujeras")
        print("  3. Presiona ESPACIO para capturar esa combinación")
        print("  4. Repite para diferentes ángulos y posiciones")
        print("\n💡 Consejos para mejores resultados:")
        print("  • Captura 100-300 muestras variadas")
        print("  • Incluye todas las posiciones del volante")
        print("  • Múltiples capturas por ángulo (con variaciones)")
        print("  • Mantén buena iluminación")
        print("  • Simula movimientos realistas de conducción")
        print("\n📁 Archivos generados:")
        print("  • steering_data/*.npy - Datos de entrenamiento")
        print("  • steering_model.h5 - Modelo entrenado")
        print("  • steering_stats.pkl - Estadísticas de normalización")
        print("="*70)


if __name__ == "__main__":
    main()
