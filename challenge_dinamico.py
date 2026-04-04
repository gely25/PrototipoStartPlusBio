import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import random
import time

# NUEVO: Añadimos utilidades de dibujo
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class DynamicChallenge:
    def __init__(self):
        self.challenges = ["IZQUIERDA", "DERECHA", "ARRIBA"]
        self.current = random.choice(self.challenges)
        self.status = "PENDIENTE"

    def check(self, landmarks):
        nose = landmarks[1]
        left_edge = landmarks[234]
        right_edge = landmarks[454]
        forehead = landmarks[10]

        # Ajuste de sensibilidad a 0.05 para que sea más fácil
        if self.current == "IZQUIERDA" and nose.x < left_edge.x + 0.05:
            return True
        if self.current == "DERECHA" and nose.x > right_edge.x - 0.05:
            return True
        if self.current == "ARRIBA" and nose.y < forehead.y + 0.12:
            return True
        return False

cap = cv2.VideoCapture(0)
challenge = DynamicChallenge()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # UI
    color = (0, 255, 255) if challenge.status == "PENDIENTE" else (0, 255, 0)
    cv2.putText(frame, f"RETO: {challenge.current}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # NUEVO: Dibujar la malla de puntos en la cara
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            points = face_landmarks.landmark
            if challenge.status == "PENDIENTE" and challenge.check(points):
                challenge.status = "COMPLETADO"

    if challenge.status == "COMPLETADO":
        cv2.putText(frame, "ACCESO CONCEDIDO", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('STARPULSE-AI Prototipo Angely', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()