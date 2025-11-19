"""
Utilidades para la interfaz de usuario del volante virtual
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont 
from .config import STEERING_BAR_Y_OFFSET, STEERING_BAR_MARGIN


def create_f1_background(width, height):
    """
    Crea un fondo metálico estilo Fórmula 1.
    """
    bg = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(bg)

    # Degradado vertical
    for y in range(height):
        shade = int(20 + (y / height) * 30)
        draw.line([(0, y), (width, y)], fill=(shade, shade, shade))

    # Borde
    draw.rectangle([(0, 0), (width - 1, height - 1)],
                   outline=(60, 60, 60), width=4)

    return bg

def draw_f1_light(pil_img, center, radius, is_on):
    """
    Dibuja una luz realista de F1 con glow.
    """
    draw = ImageDraw.Draw(pil_img)
    cx, cy = center

    if is_on:
        color_inner = (255, 40, 40)
        color_outer = (160, 0, 0)
        glow_color = (255, 60, 60)
    else:
        color_inner = (120, 120, 120)
        color_outer = (70, 70, 70)
        glow_color = None

    # Glow
    if glow_color:
        for g in range(8):
            draw.ellipse(
                (cx - radius - g, cy - radius - g,
                 cx + radius + g, cy + radius + g),
                fill=(glow_color[0], glow_color[1], glow_color[2], max(0, 80 - g * 10))
            )

    # Borde
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=color_outer,
        outline=(220, 220, 220),
        width=3
    )

    # Interior
    inner_r = int(radius * 0.65)
    draw.ellipse(
        (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
        fill=color_inner
    )

    # Reflejo
    draw.arc(
        (cx - radius + 8, cy - radius + 8,
         cx + radius - 8, cy + radius - 8),
        start=20, end=160,
        fill=(255, 255, 255),
        width=6
    )

def draw_f1_text(frame, text, y, color=(0, 255, 0), size=90):
    """
    Dibuja texto grande estilo Fórmula 1 usando Pillow
    """

    from PIL import Image, ImageDraw, ImageFont

    # Convertir frame OpenCV → PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Intenta cargar fuente F1, si no existe usa una por defecto
    try:
        font = ImageFont.truetype("Assets/F1Font.ttf", size)
    except:
        font = ImageFont.truetype("arial.ttf", size)

    # Obtener caja delimitadora del texto (compatible con versiones antiguas)
    bbox = draw.textbbox((0, 0), text, font=font)
    w_text = bbox[2] - bbox[0]
    h_text = bbox[3] - bbox[1]

    # Centrar horizontalmente
    x = (frame.shape[1] - w_text) // 2

    # Dibujar sombra
    draw.text((x + 4, y + 4), text, font=font, fill=(0, 0, 0))

    # Texto principal
    draw.text((x, y), text, font=font, fill=color)

    # Convertir atrás a OpenCV
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return frame

def draw_f1_countdown(frame, lights_on):
    """
    Semáforo estilo Fórmula 1 realista con fondo metálico, luces con glow y texto F1.
    """
    h, w = frame.shape[:2]

    # Tamaño del módulo del semáforo
    box_w = int(w * 0.80)
    box_h = int(h * 0.18)

    x0 = (w - box_w) // 2
    y0 = 10

    # Crear fondo PIL
    bg = create_f1_background(box_w, box_h)

    # Luz
    total_lights = 5
    spacing = box_w // (total_lights + 1)
    radius = int(box_h * 0.28)
    cy = box_h // 2

    for i in range(total_lights):
        cx = spacing * (i + 1)
        draw_f1_light(bg, (cx, cy), radius, i < lights_on)

    # Convertir y colocar
    bg_np = np.array(bg)
    bg_cv = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
    frame[y0:y0 + box_h, x0:x0 + box_w] = bg_cv

    # Texto
    if lights_on > 0:
        text = "PREPÁRATE"
        color = (255, 200, 0)
    else:
        text = "GO! GO! GO!"
        color = (0, 255, 0)

    text_y = y0 + box_h + 40
    frame = draw_f1_text(frame, text, text_y, color=color, size=90)

    return frame

def draw_steering_interface(frame, angle):
    h, w = frame.shape[:2]

    bar_y = h - STEERING_BAR_Y_OFFSET
    bar_x_start = STEERING_BAR_MARGIN
    bar_x_end = w - STEERING_BAR_MARGIN
    bar_width = bar_x_end - bar_x_start

    cv2.rectangle(frame, (bar_x_start, bar_y - 20),
                  (bar_x_end, bar_y + 20), (50, 50, 50), -1)

    center_x = bar_x_start + bar_width // 2
    cv2.line(frame, (center_x, bar_y - 25),
             (center_x, bar_y + 25), (255, 255, 255), 2)

    current_x = int(center_x + (angle * bar_width / 2))
    color = get_angle_color(angle)
    cv2.circle(frame, (current_x, bar_y), 15, color, -1)

    angle_text = f"Angulo: {angle:.2f}"
    cv2.putText(frame, angle_text, (bar_x_start, bar_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    draw_labels(frame, bar_x_start, bar_x_end, center_x, bar_y)

    return frame


def get_angle_color(angle):
    if abs(angle) < 0.3:
        return (0, 255, 0)
    elif abs(angle) < 0.7:
        return (0, 165, 255)
    else:
        return (0, 0, 255)


def draw_labels(frame, bar_x_start, bar_x_end, center_x, bar_y):
    cv2.putText(frame, "-1.0 (IZQ)", (bar_x_start - 30, bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "0.0", (center_x - 15, bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "+1.0 (DER)", (bar_x_end - 70, bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_direction_indicator(frame, angle, frame_width):
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
    print("\n" + "=" * 70)
    print("MODO ENTRENAMIENTO - VOLANTE VIRTUAL (CAPTURA MANUAL)")
    print("=" * 70)
    print()


def print_prediction_header():
    print("\n" + "=" * 70)
    print("MODO PREDICCION - VOLANTE VIRTUAL")
    print("=" * 70)
    print()


def print_console_value(angle_smoothed, frame_count):
    if frame_count % 10 == 0:
        print(f"Volante: {angle_smoothed:+.3f}  |  ", end="")
        bar_length = 40
        center = bar_length // 2
        pos = int(center + angle_smoothed * center)
        pos = max(0, min(bar_length - 1, pos))
        bar = ['-'] * bar_length
        bar[center] = '|'
        bar[pos] = chr(9608)
        print(''.join(bar))


def run_f1_countdown(cam):
    import time

    sequence = [
        (5, 4.0),
        (4, 4.0),
        (3, 4.0),
        (2, 4.0),
        (1, 4.0),
        (0, 1.0),   # GO!
    ]

    countdown_start = time.time()
    total_duration = sum(dur for _, dur in sequence)

    GO_REACHED = False  # marcamos cuando se completa el semáforo

    for lights_on, duration in sequence:
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = cam.read()
            if not ret:
                return False

            frame = cv2.flip(frame, 1)

            #VOLANTE ANTES DEL GO (quieto)
            frame = draw_steering_wheel_visual(frame, angle=0)

            #SEMÁFORO
            frame = draw_f1_countdown(frame, lights_on)

            #TEMPORIZADOR
            total_elapsed = time.time() - countdown_start
            total_remaining = total_duration - total_elapsed

            h, w = frame.shape[:2]
            time_text = f"{total_remaining:.1f}s"
            cv2.putText(frame, time_text, (w // 2 - 40, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.imshow("Entrenamiento", frame)

            # Salir
            if cv2.waitKey(30) & 0xFF == ord('q'):
                return False

    # habilita movimiento del volante
    return True


def draw_steering_wheel_visual(frame, angle, wheel_image_cache={}):
    import os

    if 'image' not in wheel_image_cache:
        wheel_path = "Assets/steeringWheel.png"
        if os.path.exists(wheel_path):
            wheel_image_cache['image'] = cv2.imread(wheel_path, cv2.IMREAD_UNCHANGED)
        else:
            wheel_image_cache['image'] = None

    wheel_img = wheel_image_cache['image']
    if wheel_img is None:
        return frame

    wheel_size = 150
    wheel_img_resized = cv2.resize(wheel_img, (wheel_size, wheel_size))

    h, w = frame.shape[:2]
    pos_x = (w - wheel_size) // 2
    pos_y = h - wheel_size - 150

    angle_deg = angle * 90

    center = (wheel_size // 2, wheel_size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(
        wheel_img_resized, rotation_matrix, (wheel_size, wheel_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    try:
        if pos_y < 0:
            pos_y = 0
        if pos_y + wheel_size > h:
            wheel_size = h - pos_y
            rotated = rotated[:wheel_size, :]
        if pos_x + wheel_size > w:
            wheel_size_x = w - pos_x
            rotated = rotated[:, :wheel_size_x]

        h_wheel, w_wheel = rotated.shape[:2]

        if rotated.shape[2] == 4:
            alpha = rotated[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)

            overlay = rotated[:, :, :3]
            background = frame[pos_y:pos_y + h_wheel, pos_x:pos_x + w_wheel]

            frame[pos_y:pos_y + h_wheel, pos_x:pos_x + w_wheel] = (
                alpha * overlay + (1 - alpha) * background
            ).astype(np.uint8)
        else:
            frame[pos_y:pos_y + h_wheel, pos_x:pos_x + w_wheel] = rotated[:, :, :3]

    except:
        pass

    return frame
