"""
Modelo de Deep Learning para predicción de ángulo de volante
"""

import numpy as np
import os
import pickle
from tensorflow import keras
from tensorflow.keras import layers, regularizers
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
        Construye red neuronal avanzada para regresión
        
        Arquitectura optimizada:
        - 5 capas densas con conexiones residuales
        - BatchNormalization para estabilidad
        - Dropout para regularización
        - Regularización L2
        - Huber loss (robusto ante outliers)
        - Learning rate scheduling
        
        Returns:
            model: Modelo compilado de Keras
        """
        inputs = layers.Input(shape=(self.input_size,))
        
        # Normalización inicial
        x = layers.BatchNormalization()(inputs)
        
        # Bloque 1: 256 neuronas - Extracción de características
        x1 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Dropout(0.4)(x1)
        
        # Bloque 2: 256 neuronas con conexión residual
        x2 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        # Conexión residual
        x2_residual = layers.Add()([x1, x2])
        
        # Bloque 3: 128 neuronas - Combinación de características
        x3 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x2_residual)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Activation('relu')(x3)
        x3 = layers.Dropout(0.3)(x3)
        
        # Bloque 4: 64 neuronas - Refinamiento
        x4 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x3)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.Activation('relu')(x4)
        x4 = layers.Dropout(0.2)(x4)
        
        # Bloque 5: 32 neuronas
        x5 = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))(x4)
        x5 = layers.Activation('relu')(x5)
        x5 = layers.Dropout(0.2)(x5)
        
        # Salida - regresión con tanh para rango [-1, 1]
        outputs = layers.Dense(1, activation='tanh')(x5)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Learning rate scheduling para mejor convergencia
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        
        # Compilar con Huber loss (más robusto que MSE)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
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
        
        # Callbacks (sin ReduceLROnPlateau porque usamos LearningRateSchedule)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
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
        
        # Evaluar modelo
        val_loss, val_mae, val_mse = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS FINALES:")
        print(f"  Loss (Huber): {val_loss:.4f}")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  MSE: {val_mse:.4f}")
        print(f"  Precisión estimada: +/- {val_mae:.2f} en rango [-1, 1]")
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
        
        # Cargar modelo sin compilar
        self.model = keras.models.load_model(model_path, compile=False)
        
        # Re-compilar con learning rate scheduling y Huber loss
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
        )
        
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
        
        print(f"OK Modelo cargado desde {model_path}")
    
    def get_training_metrics(self):
        """
        Obtiene las métricas del último entrenamiento
        
        Returns:
            dict: Diccionario con métricas de entrenamiento y validación
        """
        if self.history is None:
            return None
        
        metrics = {
            'train_loss': self.history.history['loss'][-1] if 'loss' in self.history.history else None,
            'train_mae': self.history.history['mae'][-1] if 'mae' in self.history.history else None,
            'val_loss': self.history.history['val_loss'][-1] if 'val_loss' in self.history.history else None,
            'val_mae': self.history.history['val_mae'][-1] if 'val_mae' in self.history.history else None,
            'epochs_trained': len(self.history.history['loss']) if 'loss' in self.history.history else 0
        }
        
        return metrics
    
    def print_model_summary(self):
        """
        Imprime un resumen de la calidad del modelo
        """
        if self.model is None:
            print("ERROR: No hay modelo cargado")
            return
        
        print("\n" + "="*70)
        print("EVALUACIÓN DEL MODELO")
        print("="*70)
        
        # Información básica del modelo
        total_params = self.model.count_params()
        print(f"Parámetros totales: {total_params:,}")
        
        # Métricas de entrenamiento si existen
        metrics = self.get_training_metrics()
        if metrics and metrics['train_loss'] is not None:
            print(f"\nMétricas de Entrenamiento:")
            print(f"   ├─ Épocas completadas: {metrics['epochs_trained']}")
            print(f"   ├─ Loss (Entrenamiento): {metrics['train_loss']:.6f}")
            print(f"   ├─ Loss (Validación): {metrics['val_loss']:.6f}")
            print(f"   ├─ MAE (Entrenamiento): {metrics['train_mae']:.6f}")
            print(f"   └─ MAE (Validación): {metrics['val_mae']:.6f}")
            
            # Análisis de calidad
            val_mae = metrics['val_mae']
            loss_diff = abs(metrics['train_loss'] - metrics['val_loss'])
            
            print(f"\nAnálisis de Calidad:")
            
            # Precisión
            if val_mae < 0.05:
                quality = "EXCELENTE"
            elif val_mae < 0.10:
                quality = "BUENA"
            elif val_mae < 0.20:
                quality = "ACEPTABLE"
            else:
                quality = "NECESITA MEJORA"
            
            print(f"   ├─ Precisión: {quality}")
            print(f"   ├─ Error promedio: ±{val_mae:.4f} (rango [-1, 1])")
            
            # Overfitting
            if loss_diff > 0.05:
                print(f"   └─ ADVERTENCIA: Posible overfitting detectado")
                print(f"      (Diferencia train-val: {loss_diff:.6f})")
            else:
                print(f"   └─ Sin signos de overfitting")
        else:
            print("\nEl modelo aún no ha sido entrenado o fue cargado desde archivo.")
        
        print("="*70 + "\n")
    
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
