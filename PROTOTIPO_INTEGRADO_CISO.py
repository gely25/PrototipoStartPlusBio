import cv2
import mediapipe as mp
import random
import time
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# --- CONFIGURACIÓN DE LIBRERÍAS (ESTILO LAURA) ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Optimizamos FaceMesh para el Challenge 3 (Antispoofing)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, # CLAVE: Detecta profundidad de ojos y nariz
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- CLASE INTEGRADA STARPULSE (TU MEJORA) ---
class StarPulseApp:
    def __init__(self):
        # Unimos los retos de Laura con tu lógica aleatoria
        self.retos_disponibles = ["GIRAR A LA IZQUIERDA", "GIRAR A LA DERECHA", "MIRAR HACIA ARRIBA"]
        self.reto_actual = random.choice(self.retos_disponibles)
        self.estado = "ESCANEO" # ESCANEO, EXITOSO, ERROR
        self.tiempo_inicio = time.time()
        self.timeout = 15 # 15 segundos de margen de seguridad

    def verificar_presencia_real(self, puntos):
        """ Esta es tu aportación técnica al proyecto de Laura """
        nariz = puntos[1]
        borde_izq = puntos[234]
        borde_der = puntos[454]
        frente = puntos[10]

        # Lógica de detección de movimiento 3D (Antispoofing)
        if self.reto_actual == "GIRAR A LA IZQUIERDA" and nariz.x < borde_izq.x + 0.08:
            return True
        if self.reto_actual == "GIRAR A LA DERECHA" and nariz.x > borde_der.x - 0.08:
            return True
        if self.reto_actual == "MIRAR HACIA ARRIBA" and nariz.y < frente.y + 0.12:
            return True
        return False

# --- INICIO DEL PROTOTIPO ---
cap = cv2.VideoCapture(0)
app = StarPulseApp()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # 1. DISEÑO DE INTERFAZ (UI STARPULSE)
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(frame, "STARPULSE-AI | BIOMETRIC AUTH v2.0", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    tiempo_restante = int(app.timeout - (time.time() - app.tiempo_inicio))

    # 2. LÓGICA DE ESTADOS
    if app.estado == "ESCANEO":
        cv2.putText(frame, f"RETO: {app.reto_actual}", (w//4, 120), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"TIEMPO: {tiempo_restante}s", (w-150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if tiempo_restante <= 0: app.estado = "ERROR"

    # 3. PROCESAMIENTO DE MALLA FACIAL
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujamos la malla como en el doc de Laura
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                connection_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Validamos tu challenge
            if app.estado == "ESCANEO" and app.verificar_presencia_real(face_landmarks.landmark):
                app.estado = "EXITOSO"

    # 4. PANTALLAS DE RESULTADO (DECISIÓN CISO)
    if app.estado == "EXITOSO":
        cv2.rectangle(frame, (w//4, h//2-40), (3*w//4, h//2+40), (0, 200, 0), -1)
        cv2.putText(frame, "ACCESO CONCEDIDO", (w//4+20, h//2+10), 2, 1, (255, 255, 255), 2)
    elif app.estado == "ERROR":
        cv2.rectangle(frame, (w//4, h//2-40), (3*w//4, h//2+40), (0, 0, 200), -1)
        cv2.putText(frame, "FALLO DE SEGURIDAD", (w//4+15, h//2+10), 2, 0.8, (255, 255, 255), 2)

    cv2.imshow('StarPulse Prototipo Integrado', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()