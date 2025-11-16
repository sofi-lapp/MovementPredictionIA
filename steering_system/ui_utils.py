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
    
    # Dirección (comentado - no mostrar en ventana)
    # draw_direction_indicator(frame, angle, w)
    
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


def draw_f1_countdown(frame, lights_on):
    """
    Dibuja semáforo estilo Fórmula 1 con 5 luces rojas
    
    Args:
        frame: Frame de video
        lights_on: Número de luces rojas encendidas (0-5)
    
    Returns:
        frame: Frame con el semáforo dibujado
    """
    h, w = frame.shape[:2]
    
    # Configuración del semáforo
    light_radius = 40
    light_spacing = 100
    lights_total = 5
    
    # Calcular posición inicial (centrado horizontal)
    total_width = (lights_total - 1) * light_spacing
    start_x = (w - total_width) // 2
    y_pos = h // 3  # Tercio superior
    
    # Fondo negro del semáforo
    margin = 30
    cv2.rectangle(frame, 
                  (start_x - margin - light_radius, y_pos - light_radius - margin),
                  (start_x + total_width + margin + light_radius, y_pos + light_radius + margin),
                  (20, 20, 20), -1)
    cv2.rectangle(frame, 
                  (start_x - margin - light_radius, y_pos - light_radius - margin),
                  (start_x + total_width + margin + light_radius, y_pos + light_radius + margin),
                  (100, 100, 100), 3)
    
    # Dibujar las 5 luces
    for i in range(lights_total):
        x_pos = start_x + i * light_spacing
        
        # Luz encendida (roja) o apagada (gris oscuro)
        if i < lights_on:
            color = (0, 0, 255)  # Rojo brillante
            cv2.circle(frame, (x_pos, y_pos), light_radius, color, -1)
            # Efecto de brillo
            cv2.circle(frame, (x_pos, y_pos), light_radius + 5, (0, 0, 200), 2)
        else:
            color = (50, 50, 50)  # Gris oscuro (apagada)
            cv2.circle(frame, (x_pos, y_pos), light_radius, color, -1)
        
        # Borde de la luz
        cv2.circle(frame, (x_pos, y_pos), light_radius, (200, 200, 200), 2)
    
    # Texto informativo
    if lights_on > 0:
        text = "PREPARATE..."
        color = (0, 165, 255)  # Naranja
    else:
        text = "GO! GO! GO!"
        color = (0, 255, 0)  # Verde
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    text_x = (w - text_size[0]) // 2
    text_y = y_pos + light_radius + 100
    
    # Sombra del texto
    cv2.putText(frame, text, (text_x + 3, text_y + 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    # Texto principal
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    
    return frame


def run_f1_countdown(cam):
    """
    Ejecuta la cuenta regresiva del semáforo F1 de 20 segundos
    
    Args:
        cam: Objeto de captura de video (cv2.VideoCapture)
    
    Returns:
        bool: True si se completó, False si se canceló con 'q'
    """
    import time
    
    print("\n" + "="*70)
    print("SEMAFORO DE INICIO - ESTILO FORMULA 1")
    print("="*70)
    print("\nSecuencia de 20 segundos:")
    print("  0-4s:   █████ (5 luces rojas)")
    print("  4-8s:   ████░ (4 luces rojas)")
    print("  8-12s:  ███░░ (3 luces rojas)")
    print("  12-16s: ██░░░ (2 luces rojas)")
    print("  16-20s: █░░░░ (1 luz roja)")
    print("  20s:    ░░░░░ GO! - Inicia captura")
    print("\nPresiona 'q' para cancelar")
    print("="*70 + "\n")
    
    # Secuencia: 5 luces por 4 segundos cada una
    sequence = [
        (5, 4.0),  # 5 luces por 4 segundos
        (4, 4.0),  # 4 luces por 4 segundos
        (3, 4.0),  # 3 luces por 4 segundos
        (2, 4.0),  # 2 luces por 4 segundos
        (1, 4.0),  # 1 luz por 4 segundos
        (0, 1.0),  # GO! por 1 segundo
    ]
    
    countdown_start = time.time()
    total_duration = sum(dur for _, dur in sequence)
    
    for lights_on, duration in sequence:
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cam.read()
            if not ret:
                return False
            
            frame = cv2.flip(frame, 1)
            
            # Dibujar semáforo
            frame = draw_f1_countdown(frame, lights_on)
            
            # Mostrar tiempo restante total (desde 20s hasta 0s)
            total_elapsed = time.time() - countdown_start
            total_remaining = total_duration - total_elapsed
            
            h, w = frame.shape[:2]
            time_text = f"{total_remaining:.1f}s"
            cv2.putText(frame, time_text, (w // 2 - 40, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            cv2.imshow("Entrenamiento", frame)
            
            # Check para cancelar
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("\nCuenta regresiva cancelada.")
                return False
        
        # Mensaje en consola
        if lights_on > 0:
            print(f"⚫ {lights_on} {'luces' if lights_on > 1 else 'luz'} encendida{'s' if lights_on > 1 else ''}")
        else:
            print("🏁 ¡GO! ¡Comenzando captura!\n")
    
    return True


def draw_steering_wheel_visual(frame, angle, wheel_image_cache={}):
    """
    Dibuja la imagen del volante rotada en el centro-inferior del frame
    
    Args:
        frame: Frame de video
        angle: Ángulo actual del volante (-1 a +1)
        wheel_image_cache: Diccionario para cachear la imagen cargada
    
    Returns:
        frame: Frame con el volante dibujado
    """
    import os
    
    # Cargar imagen solo una vez (caché)
    if 'image' not in wheel_image_cache:
        wheel_path = "Assets/steeringWheel.png"
        if os.path.exists(wheel_path):
            wheel_image_cache['image'] = cv2.imread(wheel_path, cv2.IMREAD_UNCHANGED)
        else:
            wheel_image_cache['image'] = None
            print(f"Advertencia: No se encontró {wheel_path}")
    
    wheel_img = wheel_image_cache['image']
    if wheel_img is None:
        return frame
    
    # Tamaño del volante
    wheel_size = 150
    wheel_img_resized = cv2.resize(wheel_img, (wheel_size, wheel_size))
    
    # Posición: centro horizontal, inferior del frame
    h, w = frame.shape[:2]
    pos_x = (w - wheel_size) // 2
    pos_y = h - wheel_size - 150  # 150 píxeles desde el fondo
    
    # Convertir ángulo de -1..+1 a grados (90° máximo en cada dirección)
    angle_deg = angle * 90  # 90 grados = cuarto de vuelta
    
    # Rotar imagen
    center = (wheel_size // 2, wheel_size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(wheel_img_resized, rotation_matrix, (wheel_size, wheel_size),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    
    # Superponer en el frame
    try:
        # Verificar límites
        if pos_y < 0:
            pos_y = 0
        if pos_y + wheel_size > h:
            wheel_size = h - pos_y
            rotated = rotated[:wheel_size, :]
        if pos_x + wheel_size > w:
            wheel_size_x = w - pos_x
            rotated = rotated[:, :wheel_size_x]
        
        if wheel_size <= 0:
            return frame
        
        h_wheel, w_wheel = rotated.shape[:2]
        
        # Superponer con transparencia si tiene canal alpha
        if rotated.shape[2] == 4:
            alpha = rotated[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            overlay = rotated[:, :, :3]
            background = frame[pos_y:pos_y+h_wheel, pos_x:pos_x+w_wheel]
            
            frame[pos_y:pos_y+h_wheel, pos_x:pos_x+w_wheel] = (
                alpha * overlay + (1 - alpha) * background
            ).astype(np.uint8)
        else:
            # Sin transparencia - copiar directamente
            frame[pos_y:pos_y+h_wheel, pos_x:pos_x+w_wheel] = rotated[:, :, :3]
    
    except Exception as e:
        pass  # Silenciar errores de superposición
    
    return frame
