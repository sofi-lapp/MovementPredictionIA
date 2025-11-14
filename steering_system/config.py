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
