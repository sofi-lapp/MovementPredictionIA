import cv2
import mediapipe as mp
 
# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
 
hands = mp_hands.Hands(
    max_num_hands=2,         # Detecta hasta 2 manos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
 
# Abrir cámara
cam = cv2.VideoCapture(0)
 
while True:
    ret, frame = cam.read()
    if not ret:
        break
 
    # Convertir imagen a RGB (MediaPipe usa RGB, OpenCV usa BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
 
    # Dibujar manos si se detectan
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
    # Mostrar video
    cv2.imshow("Detección de Manos", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cam.release()
cv2.destroyAllWindows()