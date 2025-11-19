"""
Configuración global del sistema de volante virtual
"""

# Directorios
DATA_DIR = "steering_data"
MODEL_DIR = "models"

# Modelo
INPUT_SIZE = 126  # 2 manos × 21 landmarks × 3 coords
MODEL_PATH = f"{MODEL_DIR}/steering_model.h5"
STATS_PATH = f"{MODEL_DIR}/steering_stats.pkl"

# MediaPipe
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Entrenamiento
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Predicción en tiempo real
PREDICTION_HISTORY_SIZE = 5
CONSOLE_UPDATE_INTERVAL = 10  # frames

# UI
STEERING_BAR_Y_OFFSET = 80
STEERING_BAR_MARGIN = 50

# Distribución de ángulos
TOTAL_ANGLES = 41  # 20 izq + centro + 20 der (-1.0 a +1.0 en pasos de 0.05)


def calculate_angle_distribution(num_samples):
    """
    Genera una distribución de cuántas muestras deben capturarse para cada ángulo.

    Estrategia:
    - Centro (0.0): Mayor cantidad de muestras
    - Extremos (-1.0, +1.0): Segunda mayor cantidad
    - Posiciones intermedias: Menos muestras

    Además calcula cuántas "vueltas completas" se darían.
    """
    import numpy as np
    
    # Definir todos los ángulos posibles
    angles = np.linspace(-1.0, 1.0, TOTAL_ANGLES)
    
    # Calcular pesos (prioridad) para cada ángulo
    # Centro = peso alto, extremos = peso medio-alto, intermedios = peso bajo
    weights = []
    for angle in angles:
        abs_angle = abs(angle)
        
        if abs_angle < 0.05:  # Centro
            weight = 5.0  # Máxima prioridad
        elif abs_angle > 0.95:  # Extremos
            weight = 3.5  # Alta prioridad
        elif abs_angle > 0.75:  # Cerca de extremos
            weight = 2.0
        elif abs_angle < 0.25:  # Cerca del centro
            weight = 2.5
        else:  # Intermedios
            weight = 1.0
        
        weights.append(weight)
    
    # Normalizar pesos para que sumen num_samples
    weights = np.array(weights)
    weights = weights / weights.sum() * num_samples
    
    # Convertir a enteros (al menos 1 muestra por ángulo)
    samples_per_angle = np.maximum(1, np.round(weights).astype(int))
    
    # Ajustar para que la suma sea exactamente num_samples
    difference = num_samples - samples_per_angle.sum()
    
    if difference > 0:
        # Agregar muestras faltantes priorizando centro y extremos
        priority_indices = [20]  # Centro
        priority_indices.extend([0, 40])  # Extremos
        for i in range(abs(difference)):
            samples_per_angle[priority_indices[i % len(priority_indices)]] += 1
    elif difference < 0:
        # Quitar muestras sobrantes de posiciones intermedias
        for i in range(abs(difference)):
            # Buscar índices intermedios con más de 1 muestra
            mid_indices = [i for i in range(5, 36) if i not in [20] and samples_per_angle[i] > 1]
            if mid_indices:
                samples_per_angle[mid_indices[i % len(mid_indices)]] -= 1
    
    # Crear diccionario de distribución
    distribution = {float(angle): int(count) for angle, count in zip(angles, samples_per_angle)}
    
    # Calcular número de vueltas (cada vuelta cubre todos los ángulos una vez)
    max_samples_per_angle = int(samples_per_angle.max())
    num_laps = max_samples_per_angle
    
    return distribution, num_laps
