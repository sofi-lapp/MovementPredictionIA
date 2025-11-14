"""
Utilidades para la interfaz de usuario del volante virtual
"""

import cv2
import numpy as np
from .config import STEERING_BAR_Y_OFFSET, STEERING_BAR_MARGIN


def draw_steering_interface(frame, angle):
    """
    Dibuja interfaz de volante virtual en el frame
    
    Args:
        frame: Frame de video
        angle: Ángulo actual del volante (-1 a +1)
    
    Returns:
        frame: Frame con interfaz dibujada
    """
    h, w = frame.shape[:2]
    
    # Barra de ángulo
    bar_y = h - STEERING_BAR_Y_OFFSET
    bar_x_start = STEERING_BAR_MARGIN
    bar_x_end = w - STEERING_BAR_MARGIN
    bar_width = bar_x_end - bar_x_start
    
    # Fondo de la barra
    cv2.rectangle(frame, (bar_x_start, bar_y - 20), 
                  (bar_x_end, bar_y + 20), (50, 50, 50), -1)
    
    # Centro
    center_x = bar_x_start + bar_width // 2
    cv2.line(frame, (center_x, bar_y - 25), 
             (center_x, bar_y + 25), (255, 255, 255), 2)
    
    # Indicador de posición actual
    current_x = int(center_x + (angle * bar_width / 2))
    color = get_angle_color(angle)
    cv2.circle(frame, (current_x, bar_y), 15, color, -1)
    
    # Texto de ángulo
    angle_text = f"Angulo: {angle:.2f}"
    cv2.putText(frame, angle_text, (bar_x_start, bar_y - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Etiquetas
    draw_labels(frame, bar_x_start, bar_x_end, center_x, bar_y)
    
    # Dirección
    draw_direction_indicator(frame, angle, w)
    
    return frame


def get_angle_color(angle):
    """
    Retorna color según el ángulo
    
    Args:
        angle: Ángulo del volante (-1 a +1)
    
    Returns:
        tuple: Color BGR
    """
    if abs(angle) < 0.3:
        return (0, 255, 0)  # Verde - centro
    elif abs(angle) < 0.7:
        return (0, 165, 255)  # Naranja
    else:
        return (0, 0, 255)  # Rojo - extremo


def draw_labels(frame, bar_x_start, bar_x_end, center_x, bar_y):
    """
    Dibuja etiquetas de la barra de volante
    
    Args:
        frame: Frame de video
        bar_x_start: Posición X inicial de la barra
        bar_x_end: Posición X final de la barra
        center_x: Posición X del centro
        bar_y: Posición Y de la barra
    """
    cv2.putText(frame, "-1.0 (IZQ)", (bar_x_start - 30, bar_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "0.0", (center_x - 15, bar_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "+1.0 (DER)", (bar_x_end - 70, bar_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_direction_indicator(frame, angle, frame_width):
    """
    Dibuja indicador de dirección en la parte superior
    
    Args:
        frame: Frame de video
        angle: Ángulo del volante
        frame_width: Ancho del frame
    """
    if angle < -0.1:
        direction = "<- IZQUIERDA"
        dir_color = (0, 255, 255)
    elif angle > 0.1:
        direction = "DERECHA ->"
        dir_color = (255, 255, 0)
    else:
        direction = "^ RECTO"
        dir_color = (0, 255, 0)
    
    cv2.putText(frame, direction, (frame_width // 2 - 150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, dir_color, 3)


def print_training_instructions():
    """Imprime instrucciones para el modo entrenamiento"""
    print("\n" + "="*70)
    print("MODO ENTRENAMIENTO - VOLANTE VIRTUAL (CAPTURA MANUAL)")
    print("="*70)
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
    print("="*70 + "\n")


def print_prediction_header():
    """Imprime encabezado del modo predicción"""
    print("\n" + "="*70)
    print("MODO PREDICCION - VOLANTE VIRTUAL")
    print("="*70)
    print("\nControla el volante con tus manos!")
    print("Valores: -1.0 (izquierda) ... 0.0 (centro) ... +1.0 (derecha)")
    print("\nPresiona 'q' para salir\n")
    print("="*70 + "\n")


def print_console_value(angle_smoothed, frame_count):
    """
    Imprime valor del volante en consola con barra visual
    
    Args:
        angle_smoothed: Ángulo suavizado del volante
        frame_count: Contador de frames para control de frecuencia
    """
    if frame_count % 10 == 0:  # Cada 10 frames
        print(f"Volante: {angle_smoothed:+.3f}  |  ", end="")
        
        # Barra visual en consola
        bar_length = 40
        center = bar_length // 2
        pos = int(center + angle_smoothed * center)
        pos = max(0, min(bar_length - 1, pos))  # Clamp
        bar = ['-'] * bar_length
        bar[center] = '|'
        bar[pos] = chr(9608)  # █
        print(''.join(bar))
