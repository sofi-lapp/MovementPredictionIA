"""
Modelo CNN para Predicción de Movimientos de Manos
===================================================

Este modelo utiliza una Red Neuronal Convolucional (CNN) para:
1. Capturar secuencias de landmarks de manos detectadas por MediaPipe
2. Entrenar un modelo que reconozca patrones de movimiento
3. Predecir movimientos futuros basándose en secuencias temporales

Cómo funciona:
--------------
1. CAPTURA DE DATOS:
   - MediaPipe detecta 21 landmarks por mano (coordenadas x, y, z)
   - Se capturan secuencias temporales de N frames
   - Cada secuencia representa un movimiento específico (ej: deslizar, girar, etc.)

2. ARQUITECTURA DEL MODELO:
   - CNN + LSTM: Combina convolución espacial con memoria temporal
   - Las capas CNN extraen características espaciales de los landmarks
   - Las capas LSTM capturan patrones temporales en la secuencia
   - Capa densa final para clasificación de movimientos

3. ENTRENAMIENTO:
   - Se recopilan múltiples ejemplos de cada tipo de movimiento
   - El modelo aprende a distinguir entre diferentes gestos
   - Validación cruzada para evitar overfitting

4. PREDICCIÓN:
   - Dada una secuencia parcial, predice el siguiente estado
   - Puede anticipar la intención del usuario
   - Útil para interfaces interactivas en tiempo real
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
from sklearn.preprocessing import LabelEncoder
import pickle


class MovementDataCollector:
    """Clase para recopilar datos de movimientos de manos"""
    
    def __init__(self, sequence_length=30):
        """
        Args:
            sequence_length: Número de frames en cada secuencia (default: 30 frames)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.sequence_length = sequence_length
        self.data_dir = "movement_data"
        
        # Crear directorio para datos si no existe
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def extract_landmarks(self, frame):
        """
        Extrae landmarks de las manos detectadas
        
        Returns:
            numpy array con shape (42, 3) para 1 mano o (84, 3) para 2 manos
            Cada landmark tiene coordenadas (x, y, z)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer coordenadas x, y, z de cada landmark
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                landmarks.extend(hand_data)
        
        # Si no se detectan manos, rellenar con ceros
        if not landmarks:
            landmarks = [0.0] * 63  # 21 landmarks × 3 coordenadas
        
        return np.array(landmarks)
    
    def collect_movement_data(self, movement_name, num_sequences=50):
        """
        Recopila datos para un tipo específico de movimiento
        
        Args:
            movement_name: Nombre del movimiento (ej: "swipe_left", "circle", "wave")
            num_sequences: Número de secuencias a recopilar
        """
        cam = cv2.VideoCapture(0)
        sequences = []
        current_sequence = []
        sequence_count = 0
        
        print(f"\n{'='*60}")
        print(f"Recopilando datos para: {movement_name}")
        print(f"Objetivo: {num_sequences} secuencias de {self.sequence_length} frames")
        print(f"{'='*60}")
        print("\nInstrucciones:")
        print("- Presiona ESPACIO para comenzar a grabar una secuencia")
        print("- Realiza el movimiento completo")
        print("- La secuencia se guardará automáticamente")
        print("- Presiona 'q' para terminar\n")
        
        recording = False
        
        while sequence_count < num_sequences:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Extraer landmarks
            landmarks = self.extract_landmarks(frame)
            
            # Dibujar manos en el frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Mostrar estado
            color = (0, 255, 0) if recording else (0, 0, 255)
            status = "GRABANDO" if recording else "ESPERANDO"
            cv2.putText(frame, f"{status} - Secuencia {sequence_count + 1}/{num_sequences}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Frames: {len(current_sequence)}/{self.sequence_length}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Movimiento: {movement_name}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Si está grabando, agregar landmarks a la secuencia
            if recording:
                current_sequence.append(landmarks)
                
                # Si se completa la secuencia
                if len(current_sequence) == self.sequence_length:
                    sequences.append(np.array(current_sequence))
                    current_sequence = []
                    recording = False
                    sequence_count += 1
                    print(f"✓ Secuencia {sequence_count} guardada")
            
            cv2.imshow("Recopilación de Datos", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording:
                recording = True
                current_sequence = []
                print(f"→ Iniciando grabación de secuencia {sequence_count + 1}...")
            elif key == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Guardar datos
        if sequences:
            filename = os.path.join(self.data_dir, f"{movement_name}.npy")
            np.save(filename, np.array(sequences))
            print(f"\n✓ {len(sequences)} secuencias guardadas en {filename}")
        
        return np.array(sequences)


class MovementCNNModel:
    """Modelo CNN para clasificación y predicción de movimientos"""
    
    def __init__(self, sequence_length=30, num_features=63):
        """
        Args:
            sequence_length: Longitud de la secuencia temporal
            num_features: Número de características por frame (21 landmarks × 3 = 63)
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
    
    def build_model(self, num_classes):
        """
        Construye la arquitectura del modelo CNN-LSTM
        
        Arquitectura:
        1. Reshape para preparar datos para Conv1D
        2. Conv1D: Extrae características espaciales
        3. MaxPooling: Reduce dimensionalidad
        4. LSTM: Captura patrones temporales
        5. Dropout: Previene overfitting
        6. Dense: Clasificación final
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # Primera capa convolucional - detecta características básicas
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            # Segunda capa convolucional - características más complejas
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            # Tercera capa convolucional
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Capas LSTM - memoria temporal para secuencias
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            
            # Capas densas finales
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_data(self, data_dir="movement_data"):
        """
        Carga todos los datos de movimientos guardados
        
        Returns:
            X: Array de secuencias
            y: Array de etiquetas
            class_names: Nombres de las clases
        """
        X = []
        y = []
        class_names = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.npy'):
                movement_name = filename[:-4]  # Remover .npy
                class_names.append(movement_name)
                
                # Cargar secuencias
                sequences = np.load(os.path.join(data_dir, filename))
                X.extend(sequences)
                y.extend([movement_name] * len(sequences))
                
                print(f"✓ Cargadas {len(sequences)} secuencias de '{movement_name}'")
        
        # Convertir a arrays numpy
        X = np.array(X)
        y = np.array(y)
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nTotal: {len(X)} secuencias, {len(class_names)} clases")
        print(f"Shape de datos: {X.shape}")
        
        return X, y_encoded, class_names
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """
        Entrena el modelo
        
        Args:
            X: Datos de entrenamiento
            y: Etiquetas
            validation_split: Porcentaje de datos para validación
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
        """
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\nDatos de entrenamiento: {X_train.shape}")
        print(f"Datos de validación: {X_val.shape}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_movement_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Entrenar
        print("\nIniciando entrenamiento...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val)
        print(f"\n{'='*60}")
        print(f"Validación Final:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_accuracy*100:.2f}%")
        print(f"{'='*60}")
        
        return self.history
    
    def save_model(self, model_path='movement_model.h5', encoder_path='label_encoder.pkl'):
        """Guarda el modelo y el codificador de etiquetas"""
        self.model.save(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Modelo guardado en {model_path}")
        print(f"✓ Encoder guardado en {encoder_path}")
    
    def load_model(self, model_path='movement_model.h5', encoder_path='label_encoder.pkl'):
        """Carga el modelo y el codificador de etiquetas"""
        self.model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"✓ Modelo cargado desde {model_path}")
    
    def predict(self, sequence):
        """
        Predice el movimiento de una secuencia
        
        Args:
            sequence: Array de shape (sequence_length, num_features)
        
        Returns:
            movement_name: Nombre del movimiento predicho
            confidence: Confianza de la predicción (0-1)
        """
        # Asegurar dimensiones correctas
        if sequence.shape != (self.sequence_length, self.num_features):
            raise ValueError(f"Secuencia debe tener shape {(self.sequence_length, self.num_features)}")
        
        # Expandir dimensiones para batch
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predecir
        predictions = self.model.predict(sequence, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Decodificar etiqueta
        movement_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return movement_name, confidence


class RealTimePredictor:
    """Clase para hacer predicciones en tiempo real"""
    
    def __init__(self, model, sequence_length=30):
        """
        Args:
            model: Modelo entrenado (MovementCNNModel)
            sequence_length: Longitud de la secuencia
        """
        self.model = model
        self.sequence_length = sequence_length
        self.current_sequence = []
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, frame):
        """Extrae landmarks (mismo método que en DataCollector)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                landmarks.extend(hand_data)
        
        if not landmarks:
            landmarks = [0.0] * 63
        
        return np.array(landmarks), results
    
    def run(self):
        """Ejecuta predicción en tiempo real"""
        cam = cv2.VideoCapture(0)
        
        print("\n{'='*60}")
        print("Predicción en Tiempo Real")
        print("{'='*60}")
        print("Presiona 'q' para salir\n")
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Extraer landmarks
            landmarks, results = self.extract_landmarks(frame)
            
            # Dibujar manos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Agregar a secuencia actual
            self.current_sequence.append(landmarks)
            
            # Mantener solo los últimos N frames
            if len(self.current_sequence) > self.sequence_length:
                self.current_sequence.pop(0)
            
            # Hacer predicción si tenemos secuencia completa
            if len(self.current_sequence) == self.sequence_length:
                sequence = np.array(self.current_sequence)
                movement, confidence = self.model.predict(sequence)
                
                # Mostrar predicción
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.putText(frame, f"Movimiento: {movement}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confianza: {confidence*100:.1f}%", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                cv2.putText(frame, f"Recopilando frames: {len(self.current_sequence)}/{self.sequence_length}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Predicción de Movimientos", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def main():
    """Función principal con flujo completo"""
    
    print("\n" + "="*60)
    print("SISTEMA DE PREDICCIÓN DE MOVIMIENTOS DE MANOS")
    print("="*60)
    print("\nOpciones:")
    print("1. Recopilar datos de movimientos")
    print("2. Entrenar modelo")
    print("3. Predicción en tiempo real")
    print("4. Flujo completo (recopilar + entrenar + predecir)")
    
    option = input("\nSelecciona una opción (1-4): ")
    
    if option == '1':
        # FASE 1: Recopilación de datos
        collector = MovementDataCollector(sequence_length=30)
        
        movements = []
        while True:
            movement_name = input("\nNombre del movimiento (o 'fin' para terminar): ")
            if movement_name.lower() == 'fin':
                break
            movements.append(movement_name)
            num_sequences = int(input(f"¿Cuántas secuencias de '{movement_name}'? (recomendado: 30-50): "))
            collector.collect_movement_data(movement_name, num_sequences)
        
        print("\n✓ Recopilación de datos completada")
    
    elif option == '2':
        # FASE 2: Entrenamiento
        model = MovementCNNModel(sequence_length=30)
        
        # Cargar datos
        X, y, class_names = model.load_data()
        
        # Construir modelo
        num_classes = len(class_names)
        model.build_model(num_classes)
        
        print("\nResumen del modelo:")
        model.model.summary()
        
        # Entrenar
        model.train(X, y, epochs=50)
        
        # Guardar
        model.save_model()
        
        print("\n✓ Entrenamiento completado")
    
    elif option == '3':
        # FASE 3: Predicción en tiempo real
        model = MovementCNNModel(sequence_length=30)
        
        try:
            model.load_model()
            predictor = RealTimePredictor(model, sequence_length=30)
            predictor.run()
        except Exception as e:
            print(f"Error: {e}")
            print("Asegúrate de haber entrenado un modelo primero (opción 2)")
    
    elif option == '4':
        # FLUJO COMPLETO
        print("\n" + "="*60)
        print("FLUJO COMPLETO")
        print("="*60)
        
        # Definir movimientos a recopilar
        movements_config = [
            ("swipe_left", 30),
            ("swipe_right", 30),
            ("wave", 30),
            ("circle", 30),
            ("stop", 30)
        ]
        
        print("\nSe recopilarán datos para los siguientes movimientos:")
        for movement, num_seq in movements_config:
            print(f"  - {movement}: {num_seq} secuencias")
        
        input("\nPresiona ENTER para continuar...")
        
        # Recopilar datos
        collector = MovementDataCollector(sequence_length=30)
        for movement_name, num_sequences in movements_config:
            collector.collect_movement_data(movement_name, num_sequences)
        
        # Entrenar
        model = MovementCNNModel(sequence_length=30)
        X, y, class_names = model.load_data()
        num_classes = len(class_names)
        model.build_model(num_classes)
        model.train(X, y, epochs=50)
        model.save_model()
        
        # Predicción en tiempo real
        print("\n¿Iniciar predicción en tiempo real? (s/n): ")
        if input().lower() == 's':
            predictor = RealTimePredictor(model, sequence_length=30)
            predictor.run()


if __name__ == "__main__":
    main()
