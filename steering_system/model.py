"""
Modelo de Deep Learning para predicción de ángulo de volante
"""

import numpy as np
import os
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from .config import (
    INPUT_SIZE, DATA_DIR, MODEL_PATH, STATS_PATH,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, VALIDATION_SPLIT, LEARNING_RATE
)


class SteeringWheelModel:
    """Modelo de regresión para predecir ángulo del volante"""
    
    def __init__(self, input_size=INPUT_SIZE):
        """
        Inicializa el modelo
        
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
        - Salida con activación tanh para valores continuos en [-1, 1]
        
        Returns:
            model: Modelo compilado de Keras
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
            
            # Salida - regresión con tanh para rango [-1, 1]
            layers.Dense(1, activation='tanh')
        ])
        
        # Compilar con MSE para regresión
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        self.model = model
        return model
    
    def load_all_training_data(self, data_dir=DATA_DIR):
        """
        Carga todos los datos de entrenamiento disponibles
        
        Args:
            data_dir: Directorio con los datos de entrenamiento
        
        Returns:
            tuple: (X, y) Arrays numpy con datos de entrada y salida
        
        Raises:
            ValueError: Si no se encuentran datos de entrenamiento
        """
        X_list = []
        y_list = []
        
        # Buscar archivos de datos
        position_files = [f for f in os.listdir(data_dir) if f.startswith('hand_positions_')]
        
        if not position_files:
            raise ValueError("No se encontraron datos de entrenamiento. Ejecuta primero el modo de recopilacion.")
        
        for pos_file in position_files:
            timestamp = pos_file.replace('hand_positions_', '').replace('.npy', '')
            angle_file = f'steering_angles_{timestamp}.npy'
            
            # Cargar datos
            X = np.load(os.path.join(data_dir, pos_file))
            y = np.load(os.path.join(data_dir, angle_file))
            
            X_list.append(X)
            y_list.append(y)
            
            print(f"OK Cargado: {pos_file} ({len(X)} muestras)")
        
        # Concatenar todos los datos
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        
        print(f"\nTotal de muestras: {len(X_all)}")
        
        return X_all, y_all
    
    def normalize_data(self, X, fit=False):
        """
        Normaliza los datos de entrada
        
        Args:
            X: Datos a normalizar
            fit: Si True, calcula media y desviación estándar
        
        Returns:
            numpy.ndarray: Datos normalizados
        """
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-8  # Evitar división por cero
        
        return (X - self.mean) / self.std
    
    def train(self, X, y, validation_split=VALIDATION_SPLIT, 
              epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
        """
        Entrena el modelo
        
        Args:
            X: Datos de entrada (posiciones de manos)
            y: Valores objetivo (ángulos del volante)
            validation_split: Porcentaje para validación
            epochs: Número de épocas
            batch_size: Tamaño del batch
        
        Returns:
            history: Historial de entrenamiento
        """
        # Normalizar datos
        X_normalized = self.normalize_data(X, fit=True)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X_normalized, y, test_size=validation_split, random_state=42
        )
        
        print(f"\n{'='*60}")
        print(f"Datos de entrenamiento: {X_train.shape}")
        print(f"Datos de validacion: {X_val.shape}")
        print(f"{'='*60}\n")
        
        # Asegurar que el directorio models existe
        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
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
                MODEL_PATH.replace('steering_model', 'best_steering_model'),
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
        print(f"  Precision estimada: +/- {val_mae:.2f} en rango [-1, 1]")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_model(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        """
        Guarda el modelo y estadísticas de normalización
        
        Args:
            model_path: Ruta donde guardar el modelo
            stats_path: Ruta donde guardar las estadísticas
        """
        # Asegurar que el directorio existe
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        stats_dir = os.path.dirname(stats_path)
        if stats_dir and not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        
        self.model.save(model_path)
        
        with open(stats_path, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'std': self.std
            }, f)
        
        print(f"OK Modelo guardado en {model_path}")
        print(f"OK Estadisticas guardadas en {stats_path}")
    
    def load_model(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        """
        Carga el modelo y estadísticas
        
        Args:
            model_path: Ruta del modelo guardado
            stats_path: Ruta de las estadísticas guardadas
        
        Raises:
            FileNotFoundError: Si no se encuentran los archivos
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontro el modelo en {model_path}")
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"No se encontraron estadisticas en {stats_path}")
        
        self.model = keras.models.load_model(model_path)
        
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
        
        print(f"OK Modelo cargado desde {model_path}")
    
    def predict(self, hand_landmarks):
        """
        Predice el ángulo del volante
        
        Args:
            hand_landmarks: Array de landmarks (126 valores)
        
        Returns:
            float: Ángulo predicho entre -1 y +1
        """
        # Normalizar
        landmarks_normalized = (hand_landmarks - self.mean) / self.std
        
        # Predecir
        landmarks_batch = np.expand_dims(landmarks_normalized, axis=0)
        prediction = self.model.predict(landmarks_batch, verbose=0)
        
        # Asegurar que esté en el rango [-1, 1]
        angle = np.clip(prediction[0][0], -1.0, 1.0)
        
        return angle
