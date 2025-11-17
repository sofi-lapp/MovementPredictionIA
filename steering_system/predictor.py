"""
Predictor en tiempo real del volante virtual
"""

import cv2
import mediapipe as mp
import numpy as np
from .config import (
    MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, 
    MIN_TRACKING_CONFIDENCE, PREDICTION_HISTORY_SIZE
)
from .ui_utils import draw_steering_interface, print_prediction_header, print_console_value


class RealTimeSteeringPredictor:
    """Predictor en tiempo real para el volante virtual"""
    
    def __init__(self, model):
        """
        Inicializa el predictor
        
        Args:
            model: Modelo entrenado (SteeringWheelModel)
        """
        self.model = model
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # Suavizado de predicciones
        self.prediction_history = [0.0] * PREDICTION_HISTORY_SIZE  # Inicializar en el centro
        self.history_size = PREDICTION_HISTORY_SIZE
    
    def extract_hand_landmarks(self, frame):
        """
        Extrae landmarks de ambas manos
        
        Args:
            frame: Frame de la cámara
        
        Returns:
            tuple: (landmarks array, mediapipe results)
        """
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
        """
        Suaviza las predicciones usando promedio móvil
        
        Args:
            new_prediction: Nueva predicción a agregar
        
        Returns:
            float: Predicción suavizada
        """
        self.prediction_history.append(new_prediction)
        
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        return np.mean(self.prediction_history)
    
    def run(self, show_console_output=True):
        """
        Ejecuta predicción en tiempo real
        
        Args:
            show_console_output: Si True, imprime valores en consola
        """
        # Mostrar evaluación del modelo antes de iniciar
        self.model.print_model_summary()
        
        cam = cv2.VideoCapture(0)
        
        print_prediction_header()
        
        if show_console_output:
            print("Valores del volante (tiempo real):")
            print("-" * 50)
        
        frame_count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Invertir imagen horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            
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
            if show_console_output:
                print_console_value(angle_smoothed, frame_count)
            
            # Dibujar interfaz
            frame = draw_steering_interface(frame, angle_smoothed)
            
            # Info adicional
            cv2.putText(frame, "Prediccion en Tiempo Real", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Volante Virtual - Prediccion", frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("Prediccion finalizada")
        print("="*70)
