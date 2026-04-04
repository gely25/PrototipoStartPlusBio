import cv2
import mediapipe as mp
import requests
import getpass
import os
import sys
import time
import random
import numpy as np

# ─────────────────────────────────────────────
#  MediaPipe Config
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
)

URL = "http://127.0.0.1:5000"

# Umbrales Biométricos
EAR_UMBRAL       = 0.015   # Parpadeo (Eye Aspect Ratio relativo)
BOCA_UMBRAL      = 0.16    # Sonrisa (Ancho de boca relativo - más alto = más estricto)
GIRO_DER_UMBRAL  = 0.05    # Giro Nariz vs Borde Derecho
GIRO_IZQ_UMBRAL  = 0.05    # Giro Nariz vs Borde Izquierdo
ARRIBA_UMBRAL    = 0.18    # Mirar arriba (más alto = más fácil - Ajustado v3.1.3)
ABAJO_FACTOR     = 0.38    # Proporción de bajada de nariz (menor = más difícil)
CALIDAD_BORROSO  = 100.0   # Varianza de Laplaciano (menor es borroso)

FRAMES_REQUERIDOS = 10      # Cuántos frames debe sostener el gesto

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDIA_DIR = os.path.join(BASE_DIR, "media")
if not os.path.exists(MEDIA_DIR):
    os.makedirs(MEDIA_DIR)

# Configuración de Poses de Registro
POSES_REGISTRO = [
    {"id": "blink",   "label": "PRUEBA DE VIDA (1/2): PARPADEE (👁️)", "cant": 0, "cond": ["parpadeo"]},
    {"id": "smile",   "label": "PRUEBA DE VIDA (2/2): SONRIA (😁)", "cant": 0, "cond": ["sonrisa"]},
    {"id": "frontal", "label": "POSICION (1/5): MIRA AL FRENTE (😐)", "cant": 5, "cond": []},
    {"id": "izq",     "label": "POSICION (2/5): GIRA A LA IZQUIERDA (↩️)", "cant": 5, "cond": ["giro_izq"], "arrow": "IZQUIERDA"},
    {"id": "der",     "label": "POSICION (3/5): GIRA A LA DERECHA (↪️)", "cant": 5, "cond": ["giro_der"], "arrow": "DERECHA"},
    {"id": "arriba",  "label": "POSICION (4/5): MIRA HACIA ARRIBA (⬆️)", "cant": 5, "cond": ["arriba"], "arrow": "ARRIBA"},
    {"id": "abajo",   "label": "POSICION (5/5): MIRA HACIA ABAJO (⬇️)", "cant": 5, "cond": ["abajo"], "arrow": "ABAJO"},
]

# ─────────────────────────────────────────────
#  Helpers servidor
# ─────────────────────────────────────────────
def nombre_existe(nombre):
    try:
        res = requests.get(f"{URL}/existe_usuario/{nombre}", timeout=5)
        return res.json().get("existe", False)
    except:
        return False

def registrar_en_servidor(nombre, password, fingerprint):
    try:
        res = requests.post(f"{URL}/registrar",
                            json={"nombre": nombre, "password": password, "fingerprint": fingerprint},
                            timeout=5)
        return res.status_code == 200
    except:
        return False

def verificar_credenciales(nombre, password):
    try:
        res = requests.post(f"{URL}/verificar_creds",
                            json={"nombre": nombre, "password": password},
                            timeout=5)
        if res.status_code == 200:
            return res.json()
        return None
    except:
        return None

def existe_cara_en_db(fingerprint):
    try:
        res = requests.post(f"{URL}/existe_cara", json={"fingerprint": fingerprint}, timeout=5)
        return res.json()
    except:
        return {"existe": False}

# ─────────────────────────────────────────────
#  Biometría Avanzada
# ─────────────────────────────────────────────

def get_face_fingerprint(landmarks):
    """
    Genera un vector de proporciones faciales basado en distancias entre landmarks.
    Invariante a escala (dividido por distancia interpupilar).
    """
    p = landmarks
    # Distancia normalizadora: distancia entre centros de ojos (aprox)
    norm = ((p[468].x - p[473].x)**2 + (p[468].y - p[473].y)**2)**0.5
    if norm == 0: norm = 1.0

    key_points = [
        (10, 152), (33, 263), (61, 291), (1, 4),  # Frente-mentón, Ojo-ojo, Boca, Nariz
        (33, 133), (263, 362), (61, 0), (291, 0), # Detalles ojos y boca
        (10, 1), (152, 1), (234, 454), (10, 234)  # Nariz a bordes, Ancho cara
    ]
    
    fingerprint = []
    for (i, j) in key_points:
        dist = ((p[i].x - p[j].x)**2 + (p[i].y - p[j].y)**2)**0.5
        fingerprint.append(dist / norm)
    
    return fingerprint

def calcular_borrosidad(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def evaluar_calidad(frame, landmarks):
    """Evalúa si la imagen es apta para registro."""
    borrosidad = calcular_borrosidad(frame)
    if borrosidad < CALIDAD_BORROSO:
        return False, f"Imagen borrosa ({int(borrosidad)})"
    
    # Verificar centrado
    nose = landmarks[1]
    if not (0.3 < nose.x < 0.7 and 0.3 < nose.y < 0.7):
        return False, "Rostro no centrado"
    
    return True, "Excelente"

def contar_dedos(hand_landmarks):
    """Cuenta cuántos dedos están levantados (MediaPipe Hands)."""
    if not hand_landmarks: return 0
    tips = [8, 12, 16, 20] # Índices de las puntas
    bases = [6, 10, 14, 18] # Índices de las falanges medias
    dedos = 0
    
    # Pulgar (requiere lógica distinta por eje X o Y)
    if hand_landmarks[4].x < hand_landmarks[3].x: dedos += 1
    
    for tip, base in zip(tips, bases):
        if hand_landmarks[tip].y < hand_landmarks[base].y: dedos += 1
    
    return dedos

def evaluar_posicion_mano(pts_face, pts_hand, zona_esperada):
    """Verifica si la mano está en la zona correcta respecto a la cara."""
    if not pts_face or not pts_hand: return False
    
    # Bounding box aproximado de la cara
    f_izq = pts_face[234].x
    f_der = pts_face[454].x
    f_abj = pts_face[152].y
    
    # Centro de la mano (muñeca o centro de landmarks)
    m_x = pts_hand[0].x
    m_y = pts_hand[0].y
    
    margen = 0.05 # Margen de separación de la cara
    
    if zona_esperada == "DERECHA":
        return m_x > f_der + margen
    if zona_esperada == "IZQUIERDA":
        return m_x < f_izq - margen
    if zona_esperada == "ABAJO":
        return m_y > f_abj + margen
    return False

def draw_target_zone(frame, zona):
    """Dibuja una caja guía donde el usuario debe poner la mano."""
    h, w = frame.shape[:2]
    color = (255, 255, 0)
    thick = 2
    
    if zona == "DERECHA":
        cv2.rectangle(frame, (w - 200, h//2 - 100), (w - 20, h//2 + 100), color, thick)
        cv2.putText(frame, "MANO AQUI", (w - 180, h//2 - 110), 1, 1, color, 1)
    elif zona == "IZQUIERDA":
        cv2.rectangle(frame, (20, h//2 - 100), (200, h//2 + 100), color, thick)
        cv2.putText(frame, "MANO AQUI", (40, h//2 - 110), 1, 1, color, 1)
    elif zona == "ABAJO":
        cv2.rectangle(frame, (w//2 - 100, h - 180), (w//2 + 100, h - 20), color, thick)
        cv2.putText(frame, "MANO AQUI", (w//2 - 80, h - 190), 1, 1, color, 1)

def detectar_acciones(face_landmarks, hand_landmarks=None):
    """Analiza todos los gestos simultáneamente."""
    acciones = {
        "parpadeo": False, "sonrisa": False, "giro_der": False, 
        "giro_izq": False, "arriba": False, "abajo": False, "dedos": 0,
        "mano_zona": None # Se evaluará en el flujo según el reto
    }
    if not face_landmarks: return acciones
    
    pts = face_landmarks.landmark
    # 1. Parpadeo
    ear = abs(pts[159].y - pts[145].y)
    if ear < EAR_UMBRAL: acciones["parpadeo"] = True
    
    # 2. Sonrisa
    boca = abs(pts[61].x - pts[291].x)
    if boca > BOCA_UMBRAL: acciones["sonrisa"] = True
    
    # 3. Giros
    nariz = pts[1]
    ref_der, ref_izq = pts[454], pts[234]
    if nariz.x > ref_der.x - GIRO_DER_UMBRAL: acciones["giro_der"] = True
    if nariz.x < ref_izq.x + GIRO_IZQ_UMBRAL: acciones["giro_izq"] = True
    
    # 4. Arriba / Abajo
    frente = pts[10]
    menton = pts[152]
    if nariz.y < frente.y + ARRIBA_UMBRAL: acciones["arriba"] = True
    if nariz.y > menton.y - (menton.y - frente.y) * ABAJO_FACTOR: acciones["abajo"] = True
    
    # 5. Dedos y Posición
    if hand_landmarks:
        acciones["dedos"] = contar_dedos(hand_landmarks.landmark)
        # La posición se valida dinámicamente en el flujo usando evaluar_posicion_mano
    
    return acciones

# ─────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────
def draw_arrow(frame, direction):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = (0, 255, 255)
    length = 100
    thick = 8
    
    if direction == "DERECHA":
        cv2.arrowedLine(frame, (cx - length, cy), (cx + length, cy), color, thick)
    elif direction == "IZQUIERDA":
        cv2.arrowedLine(frame, (cx + length, cy), (cx - length, cy), color, thick)
    elif direction == "ARRIBA":
        cv2.arrowedLine(frame, (cx, cy + length), (cx, cy - length), color, thick)
    elif direction == "ABAJO":
        cv2.arrowedLine(frame, (cx, cy - length), (cx, cy + length), color, thick)

def cuenta_regresiva(cap, mensaje, segundos=3):
    start_time = time.time()
    while time.time() - start_time < segundos:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        remanente = segundos - int(time.time() - start_time)
        
        # Filtro moderno
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 10, 5), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, mensaje.upper(), (w//2 - 250, h//2 - 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, str(remanente), (w//2 - 30, h//2 + 80),
                    cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 255, 0), 5)
        
        cv2.imshow("StarPulse Ultra", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def mostrar_progreso(frame, paso, total, subpaso_prog=0):
    h, w = frame.shape[:2]
    # Barra de fondo
    cv2.rectangle(frame, (50, h-40), (w-50, h-20), (40, 40, 40), -1)
    # Barra de progreso general
    ancho_total = w - 100
    progreso_gen = (paso + subpaso_prog) / total
    cv2.rectangle(frame, (50, h-40), (50 + int(ancho_total * progreso_gen), h-20), (0, 255, 0), -1)
    
    cv2.putText(frame, f"PASO {paso+1} DE {total}", (55, h-45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def mensaje_camara(frame, texto, color, duracion_ms=2000):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2-60), (w, h//2+60), color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.putText(frame, texto, (w//2 - len(texto)*10, h//2 + 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("StarPulse Ultra", frame)
    cv2.waitKey(duracion_ms)

# ─────────────────────────────────────────────
#  Flujo biométrico principal
# ─────────────────────────────────────────────

def flujo_biometrico(modo, nombre, fingerprint_esperado=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se encontró cámara.")
        return False, None

    if modo == "REGISTRO":
        # --- FLUJO DE REGISTRO GUIADO 3.1 (5 Capturas por pose) ---
        paso_actual = 0
        total_pasos = len(POSES_REGISTRO)
        pool_poses_capturadas = {}
        mejor_laplacian_global = -1
        mejor_fingerprint = None
        fingerprint_referencia = None  # Para asegurar que la cara no cambie
        
        frames_pose = 0
        capturas_pose_actual = 0
        mejor_frame_pose_actual = None
        mejor_lap_pose_actual = -1

        cuenta_regresiva(cap, "INICIANDO REGISTRO GUIADO")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_face = face_mesh.process(rgb)
            res_hands = hands.process(rgb)

            cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)
            cv2.putText(frame, f"StarPulse  |  REGISTRO  |  {nombre}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            if res_face.multi_face_landmarks:
                face_landmarks = res_face.multi_face_landmarks[0]
                hand_landmarks = res_hands.multi_hand_landmarks[0] if res_hands.multi_hand_landmarks else None
                acciones = detectar_acciones(face_landmarks, hand_landmarks)
                
                # Consistencia de Identidad: Evitar cambio de cara durante el proceso
                current_fp = get_face_fingerprint(face_landmarks.landmark)
                if fingerprint_referencia:
                    v1, v2 = np.array(current_fp), np.array(fingerprint_referencia)
                    sim_consist = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if sim_consist < 0.88: # Si la cara cambia bruscamente (ej: cambio de persona)
                        mensaje_camara(frame, "ERROR: DETECTADO CAMBIO DE IDENTIDAD", (0, 0, 200), 3000)
                        print(f" ⚠️ ALERTA: Identidad inconsistente ({sim_consist*100:.2f}%). Abortando registro.")
                        break

                if paso_actual < len(POSES_REGISTRO):
                    pose = POSES_REGISTRO[paso_actual]
                    cumple_pose = all(acciones[c] for c in pose["cond"])
                    
                    if cumple_pose:
                        frames_pose += 1
                        # Lógica diferenciada: Si es de vida (cant=0) o de captura (cant>0)
                        if pose["cant"] == 0:
                            if frames_pose >= 15:
                                print(f" ✅ Gesto de vida '{pose['id']}' validado.")
                                # Fijar la identidad de referencia en el primer reto exitoso
                                if not fingerprint_referencia:
                                    fingerprint_referencia = current_fp
                                    print(" 🧊 Identidad bloqueada para el resto del proceso.")
                                paso_actual += 1
                                frames_pose = 0
                                mensaje_camara(frame, "VIDA VERIFICADA", (0, 150, 0), 1000)
                        else:
                            if frames_pose >= 10:
                                if capturas_pose_actual < 5:
                                    lap = calcular_borrosidad(frame)
                                    if lap > mejor_lap_pose_actual:
                                        mejor_lap_pose_actual = lap
                                        mejor_frame_pose_actual = frame.copy()
                                        print(f" ✨ Nueva mejor imagen '{pose['id']}': Laplacian={lap:.2f}")
                                        
                                        if pose["id"] == "frontal" or lap > mejor_laplacian_global:
                                            mejor_laplacian_global = lap
                                            mejor_fingerprint = current_fp
                                            print(f" 🧬 Fingerprint actualizado: {[round(x,3) for x in mejor_fingerprint[:5]]}...")
                                    
                                    capturas_pose_actual += 1
                                    mensaje_camara(frame, f"CAPTURA {capturas_pose_actual}/5 ({pose['id']})", (200, 100, 0), 1000)
                                else:
                                    pool_poses_capturadas[pose["id"]] = mejor_frame_pose_actual
                                    print(f" ✅ Pose '{pose['id']}' completada.")
                                    paso_actual += 1
                                    frames_pose, capturas_pose_actual = 0, 0
                                    mejor_frame_pose_actual, mejor_lap_pose_actual = None, -1
                                    if paso_actual < len(POSES_REGISTRO):
                                        mensaje_camara(frame, "POSICION OK", (0, 150, 0), 1000)
                    else:
                        frames_pose = 0
                    
                    # UI Pose
                    if paso_actual < len(POSES_REGISTRO):
                        cv2.putText(frame, POSES_REGISTRO[paso_actual]["label"], (50, 110), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                        if "arrow" in POSES_REGISTRO[paso_actual]:
                            draw_arrow(frame, POSES_REGISTRO[paso_actual]["arrow"])
                    
                    prog = (capturas_pose_actual/5.0) if pose["cant"] > 0 else (frames_pose/15.0)
                    mostrar_progreso(frame, paso_actual, len(POSES_REGISTRO), prog * (1.0/len(POSES_REGISTRO)))

                # --- Finalizar una vez completados todos los pasos ---
                if paso_actual == len(POSES_REGISTRO):
                    print(" 🔍 Verificando unicidad biométrica en la base de datos...")
                    res_duplicado = existe_cara_en_db(mejor_fingerprint)
                    if res_duplicado["existe"]:
                        mensaje_camara(frame, f"ERROR: CARA YA REGISTRADA ({res_duplicado['nombre']})", (0,0,200), 4000)
                        print(f" ❌ Registro denegado: Esta cara ya pertenece a '{res_duplicado['nombre']}'.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False, None
                    else:
                        for pid, img in pool_poses_capturadas.items():
                            save_path = os.path.join(MEDIA_DIR, f"reg_{nombre}_{pid}.jpg")
                            cv2.imwrite(save_path, img)
                            print(f" 💾 Imagen guardada: {save_path}")
                        mensaje_camara(frame, "REGISTRO COMPLETADO", (0, 200, 0), 2000)
                        cap.release()
                        cv2.destroyAllWindows()
                        return True, mejor_fingerprint
            else:
                cv2.putText(frame, "NO SE DETECTA ROSTRO", (50, 110), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("StarPulse Ultra", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        return False, None

    else:
        # --- FLUJO DE LOGIN SECUENCIAL 1.0 (MANO -> CARA) ---
        RETOS_LOGIN = [
            {"label_m": "PONGA 3 DEDOS A LA DERECHA", "zona": "DERECHA", "dedos": 3, "label_f": "AHORA PARPADEE", "face": ["parpadeo"]},
            {"label_m": "PONGA 2 DEDOS A LA IZQUIERDA", "zona": "IZQUIERDA", "dedos": 2, "label_f": "AHORA SONRIA", "face": ["sonrisa"]},
            {"label_m": "PONGA 1 DEDO BAJO LA BARBILLA", "zona": "ABAJO", "dedos": 1, "label_f": "AHORA GIRE A LA DERECHA", "face": ["giro_der"], "arrow": "DERECHA"},
            {"label_m": "PONGA 3 DEDOS A LA IZQUIERDA", "zona": "IZQUIERDA", "dedos": 3, "label_f": "AHORA MIRE ARRIBA", "face": ["arriba"], "arrow": "ARRIBA"}
        ]
        
        retos_actuales = random.sample(RETOS_LOGIN, 2)
        paso_reto = 0 
        fase_reto = 0  # 0: Mano, 1: Cara
        frames_cumplidos = 0
        exito_final = False

        cuenta_regresiva(cap, "INICIANDO LOGIN SECUENCIAL")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_face = face_mesh.process(rgb)
            res_hands = hands.process(rgb)
            
            cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)
            cv2.putText(frame, f"StarPulse  |  COMPARANDO CON: {nombre}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            if res_face.multi_face_landmarks:
                face_landmarks = res_face.multi_face_landmarks[0]
                hand_landmarks = res_hands.multi_hand_landmarks[0] if res_hands.multi_hand_landmarks else None
                acciones = detectar_acciones(face_landmarks, hand_landmarks)
                
                if paso_reto < len(retos_actuales):
                    reto = retos_actuales[paso_reto]
                    
                    if fase_reto == 0:
                        cv2.putText(frame, reto["label_m"], (50, 110), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                        draw_target_zone(frame, reto["zona"])
                        
                        cumple_dedos = (acciones["dedos"] == reto["dedos"])
                        pts_hand = hand_landmarks.landmark if hand_landmarks else None
                        cumple_zona = evaluar_posicion_mano(face_landmarks.landmark, pts_hand, reto["zona"])
                        
                        if cumple_dedos and cumple_zona:
                            frames_cumplidos += 1
                        else:
                            frames_cumplidos = 0
                            
                        if frames_cumplidos >= 15:
                            fase_reto = 1
                            frames_cumplidos = 0
                            mensaje_camara(frame, "POSICION OK", (0, 150, 0), 800)
                    
                    elif fase_reto == 1:
                        cv2.putText(frame, reto["label_f"], (50, 110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                        if "arrow" in reto: draw_arrow(frame, reto["arrow"])
                        
                        if all(acciones[f] for f in reto["face"]):
                            frames_cumplidos += 1
                        else:
                            frames_cumplidos = 0
                            
                        if frames_cumplidos >= FRAMES_REQUERIDOS:
                            paso_reto += 1
                            fase_reto = 0
                            frames_cumplidos = 0
                            if paso_reto < len(retos_actuales):
                                mensaje_camara(frame, "RETO 1 COMPLETADO", (0, 150, 0), 1000)
                            else:
                                actual_fp = get_face_fingerprint(face_landmarks.landmark)
                                v1, v2 = np.array(actual_fp), np.array(fingerprint_esperado)
                                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                                print(f" 🔍 Comparando identidad: Similitud = {sim*100:.2f}%")
                                
                                if sim > 0.92:
                                    print(" ✅ Acceso concedido (Similitud > 92%)")
                                    mensaje_camara(frame, "ACCESO CONCEDIDO", (0, 200, 0), 2500)
                                    exito_final = True
                                else:
                                    print(f" ❌ Acceso denegado: Similitud {sim*100:.2f}% insuficiente.")
                                    mensaje_camara(frame, "ERROR BIOMETRICO", (0, 0, 200), 3000)
                                    exito_final = False
                                break
                    
                    mostrar_progreso(frame, paso_reto, len(retos_actuales), (fase_reto + min(frames_cumplidos/15.0, 1.0))/2.0)
            else:
                cv2.putText(frame, "NO SE DETECTA ROSTRO", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("StarPulse Ultra", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        return exito_final, None

# ─────────────────────────────────────────────
#  Menú principal
# ─────────────────────────────────────────────
def main():
    while True:
        print("\n╔══════════════════════════════════╗")
        print("║      StarPulse Ultra  –  ZK     ║")
        print("╠══════════════════════════════════╣")
        print("║  [1] Registrarse                ║")
        print("║  [2] Iniciar sesion             ║")
        print("║  [3] Salir                      ║")
        print("╚══════════════════════════════════╝")
        op = input("\nSeleccione: ").strip()

        if op == "1":
            nombre = input("Nombre de usuario: ").strip()
            if not nombre:
                print("❌ El nombre no puede estar vacío.")
                continue

            print("🔍 Verificando disponibilidad del nombre...")
            if nombre_existe(nombre):
                print(f"❌ El usuario '{nombre}' ya existe. Elija otro nombre.")
                continue

            password = getpass.getpass("Contraseña: ")
            confirm  = getpass.getpass("Confirmar contraseña: ")
            if password != confirm:
                print("❌ Las contraseñas no coinciden.")
                continue
            if len(password) < 4:
                print("❌ La contraseña debe tener al menos 4 caracteres.")
                continue

            input("\n📸 Todo listo. Presione [ENTER] para abrir la cámara e iniciar el registro...")
            print("⏳ Cargando MediaPipe...")
            exito, fingerprint = flujo_biometrico("REGISTRO", nombre)
            
            if exito:
                if registrar_en_servidor(nombre, password, fingerprint):
                    print(f"✅ Usuario '{nombre}' registrado correctamente.")
                else:
                    print("❌ Error al guardar en el servidor.")
            else:
                print("🚫 Registro cancelado o fallido.")

        elif op == "2":
            nombre   = input("Usuario: ").strip()
            password = getpass.getpass("Contraseña: ")

            print("\n🔍 Verificando credenciales...")
            res_creds = verificar_credenciales(nombre, password)
            if not res_creds:
                print("❌ Usuario o contraseña incorrectos.")
                continue

            print(f"✅ Credenciales OK. Verificando perfil biométrico de '{nombre}'.")
            input("📸 Presione [ENTER] para iniciar la comparación facial en tiempo real...")
            print(f"🧬 Descargando plantilla biométrica de '{nombre}' desde el servidor...")
            print("👁️‍🗨️ El sistema comparará tu cara contra la guardada durante el registro.")
            print("⏳ Cargando MediaPipe...")
            # Pasar el fingerprint guardado en la DB para comparar
            exito, _ = flujo_biometrico("LOGIN", nombre, fingerprint_esperado=res_creds.get("fingerprint"))
            
            if exito:
                print(f"\n🔓 BIENVENIDO, {nombre}. ACCESO TOTAL CONCEDIDO.")
            else:
                print("🚫 Verificación biométrica fallida.")

        elif op == "3":
            print("👋 Saliendo...")
            break
        else:
            print("⚠️  Opción no válida.")

if __name__ == "__main__":
    main()

# import cv2
# import mediapipe as mp
# import requests
# import time

# # Configuración MediaPipe
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

# class StarPulseSistema:
#     def __init__(self):
#         self.url = "http://127.0.0.1:5000"

#     def flujo_biometrico(self, modo="LOGIN", nombre=""):
#         cap = cv2.VideoCapture(0)
#         # Forzamos resolución para que los mensajes no se corten
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         p_blink, p_smile, p_move = False, False, False
#         capturas = 0
#         exito = False

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret: break
            
#             frame = cv2.flip(frame, 1)
#             h, w, _ = frame.shape
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = face_mesh.process(rgb)

#             # --- INTERFAZ LIMPIA ---
#             # Barra negra superior para que el texto sea legible
#             cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
#             cv2.putText(frame, f"MODO: {modo} | USUARIO: {nombre}", (20, 45), 1, 1.8, (255, 255, 255), 2)

#             if res.multi_face_landmarks:
#                 pts = res.multi_face_landmarks[0].landmark
                
#                 # RETO 1: PARPADEO (EAR)
#                 if not p_blink:
#                     if abs(pts[159].y - pts[145].y) < 0.01:
#                         p_blink = True
                
#                 # RETO 2: SONRISA (Distancia comisuras)
#                 if p_blink and not p_smile:
#                     dist_boca = abs(pts[61].x - pts[291].x)
#                     if dist_boca > 0.13: p_smile = True
                
#                 # RETO 3: GIRO (Solo en Login - Tu Challenge)
#                 if modo == "LOGIN":
#                     if p_smile and not p_move:
#                         if pts[1].x > pts[454].x - 0.05: p_move = True
#                 else:
#                     # En Registro termina al sonreír
#                     if p_smile: p_move = True

#             # --- MENSAJES DE ESTADO (Ajustados para no cortarse) ---
#             if not p_blink: 
#                 txt, col = "PASO 1: PARPADEE", (0, 255, 255)
#             elif not p_smile: 
#                 txt, col = "PASO 2: SONRIA", (0, 255, 255)
#             elif modo == "LOGIN" and not p_move: 
#                 txt, col = "PASO 3: GIRE A LA DERECHA", (255, 0, 255)
#             else: 
#                 txt, col = "¡RETO COMPLETADO!", (0, 255, 0)
            
#             cv2.putText(frame, txt, (20, 130), 1, 2.5, col, 3)

#             # --- PROCESAR RESULTADO ---
#             if p_move:
#                 if modo == "REGISTRO":
#                     capturas += 1
#                     cv2.imwrite(f"foto_{nombre}_{capturas}.jpg", frame)
#                     if capturas >= 5:
#                         # Cuadro de confirmación final
#                         cv2.rectangle(frame, (150, 250), (1130, 450), (0, 200, 0), -1)
#                         cv2.putText(frame, "REGISTRO OK - PULSE UNA TECLA PARA GUARDAR", (180, 350), 1, 1.5, (0, 0, 0), 2)
#                         cv2.imshow("STARPULSE", frame)
#                         cv2.waitKey(0) # Pausa obligatoria para que el usuario vea el éxito
#                         exito = True
#                         break
#                 else:
#                     # Login exitoso
#                     cv2.rectangle(frame, (300, 250), (980, 450), (0, 200, 0), -1)
#                     cv2.putText(frame, "ACCESO CONCEDIDO", (380, 360), 1, 2.5, (0, 0, 0), 3)
#                     cv2.imshow("STARPULSE", frame)
#                     cv2.waitKey(2000)
#                     exito = True
#                     break

#             cv2.imshow("STARPULSE", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'): break

#         cap.release()
#         cv2.destroyAllWindows()
#         return exito

#     def menu(self):
#         while True:
#             print("\n" + "="*30)
#             print(" SISTEMA BIOMETRICO STARPULSE")
#             print("="*30)
#             print("[1] Registrarse (ID Auto)\n[2] Login\n[3] Salir")
#             op = input("Seleccione: ")
            
#             if op == "1":
#                 # ID AUTOMATICO: Genera un número de 5 dígitos basado en el tiempo
#                 id_auto = str(int(time.time()))[-5:]
#                 nombre = input("Escriba su nombre: ")
#                 print(f"[*] Registrando a {nombre} con ID: {id_auto}...")
                
#                 if self.flujo_biometrico("REGISTRO", nombre):
#                     requests.post(f"{self.url}/registrar", json={"id": id_auto, "nombre": nombre})
#                     print(f"✅ ¡ÉXITO! {nombre} ha sido guardado.")
#                 else:
#                     print("❌ Registro cancelado o incompleto.")
            
#             elif op == "2":
#                 if self.flujo_biometrico("LOGIN", "VERIFICANDO"):
#                     print("🔓 Bienvenido al sistema.")
#                 else:
#                     print("🔒 Error de verificación.")
            
#             elif op == "3": break

# if __name__ == "__main__":
#     StarPulseSistema().menu()