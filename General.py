import cv2
import numpy as np
import datetime
 
def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)
 
def marcar_zonas(frame_mov, frame_original):
    frame_mov = cv2.GaussianBlur(frame_mov, (21, 21), 0)
    _, limites = cv2.threshold(frame_mov, 25, 255, cv2.THRESH_BINARY)
    limites = cv2.dilate(limites, None, iterations=2)
    contours, _ = cv2.findContours(limites.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    return frame_original
 
# Inicializar cámara
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()
 
# Leer los tres primeros cuadros
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)
 
while True:
    # Calcular zonas con movimiento
    delta = diffImg(t_minus, t, t_plus)
    frame_original = cam.read()[1]
    frame_original = cv2.resize(frame_original, (640, 480))
 
    zonas_marcadas = marcar_zonas(delta, frame_original.copy())
 
    # Mostrar video en tiempo real con las zonas marcadas
    cv2.imshow("Detección de Movimiento", zonas_marcadas)
 
    # Desplazar los frames
    t_minus = t
    t = t_plus
    t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)
 
    # Salir con la tecla "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cam.release()
cv2.destroyAllWindows()
 