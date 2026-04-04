import cv2
import mediapipe as mp
import requests
import time

# --- CONFIGURACIÓN STARPULSE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def obtener_datos_servidor():
    # Simulamos login de "Miranda" (ID 1)
    user_data = requests.get('http://127.0.0.1:5000/get_user/1').json()
    challenge_data = requests.get('http://127.0.0.1:5000/get_challenge').json()
    return user_data['user'], challenge_data

def iniciar_sistema():
    nombre, retos = obtener_datos_servidor()
    cap = cv2.VideoCapture(0)
    
    # Estados del sistema
    paso_1_parpadeo = False
    paso_2_movimiento = False
    
    print(f"[*] Iniciando sesión para: {nombre}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # UI Superior
        cv2.rectangle(frame, (0,0), (640, 60), (30,30,30), -1)
        cv2.putText(frame, f"USUARIO: {nombre} | STARPULSE-ZK", (20, 40), 1, 1.5, (255,255,255), 2)

        if results.multi_face_landmarks:
            p = results.multi_face_landmarks[0].landmark
            
            # --- CHALLENGE 1: PARPADEO (Laura) ---
            # Medimos distancia entre párpado superior e inferior (puntos 159 y 145)
            ojo_dist = abs(p[159].y - p[145].y)
            if not paso_1_parpadeo and ojo_dist < 0.012: # Si cierra el ojo
                paso_1_parpadeo = True

            # --- CHALLENGE 2: MOVIMIENTO (Samira) ---
            if paso_1_parpadeo and not paso_2_movimiento:
                # Si el reto es ARRIBA (Nariz 1 vs Frente 10)
                if p[1].y < p[10].y + 0.12:
                    paso_2_movimiento = True

            # Dibujar Malla
            mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION)

        # --- MENSAJES EN PANTALLA ---
        if not paso_1_parpadeo:
            cv2.putText(frame, "PASO 1: PARPADEE", (200, 120), 2, 0.8, (0,255,255), 2)
        elif not paso_2_movimiento:
            cv2.putText(frame, f"PASO 2: {retos['challenge_2']}", (150, 120), 2, 0.8, (0,255,255), 2)
        else:
            cv2.rectangle(frame, (100, 200), (540, 300), (0,255,0), -1)
            cv2.putText(frame, "IDENTIDAD CONFIRMADA", (130, 260), 2, 1, (255,255,255), 2)

        cv2.imshow('StarPulse Full Integration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_sistema()