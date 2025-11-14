"""
Sistema de Volante Virtual con Deep Learning
============================================

Este paquete modular permite controlar un volante virtual usando las manos
detectadas por MediaPipe. Devuelve valores entre -1 (izquierda máxima) y +1 
(derecha máxima).

Módulos:
    - config: Configuración y constantes del sistema
    - data_collector: Recopilación de datos de entrenamiento
    - model: Arquitectura y entrenamiento del modelo de Deep Learning
    - predictor: Predicción en tiempo real
    - ui_utils: Utilidades de interfaz gráfica

Características del modelo:
    - Input: Landmarks de ambas manos (21 landmarks × 3 coordenadas × 2 manos = 126 valores)
    - Output: Valor continuo entre -1 y +1 (regresión)
    - Arquitectura: Red neuronal densa con capas dropout y batch normalization
"""

from .config import *
from .data_collector import SteeringWheelDataCollector
from .model import SteeringWheelModel
from .predictor import RealTimeSteeringPredictor
from .ui_utils import (
    draw_steering_interface,
    print_training_instructions,
    print_prediction_header,
    print_console_value
)

__version__ = "1.0.0"
__author__ = "MovementPredictionIA"

__all__ = [
    # Clases principales
    'SteeringWheelDataCollector',
    'SteeringWheelModel',
    'RealTimeSteeringPredictor',
    
    # Funciones de utilidad
    'draw_steering_interface',
    'print_training_instructions',
    'print_prediction_header',
    'print_console_value',
    
    # Configuración
    'DATA_DIR',
    'MODEL_DIR',
    'INPUT_SIZE',
    'MODEL_PATH',
    'STATS_PATH',
]
