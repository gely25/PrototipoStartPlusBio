import cv2
import mediapipe as mp
import requests
import getpass
import os
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

# ─────────────────────────────────────────────
#  Umbrales Biométricos
# ─────────────────────────────────────────────
EAR_UMBRAL      = 0.025  # Subido: detects natural blinks without keeping eye fully closed
BOCA_UMBRAL     = 0.16
GIRO_DER_UMBRAL = 0.05
GIRO_IZQ_UMBRAL = 0.05
ARRIBA_UMBRAL   = 0.18
ABAJO_FACTOR    = 0.38
CALIDAD_BORROSO    = 100.0
ILUMINACION_MIN    = 60     # Brillo mínimo promedio (0-255) — muy oscuro = rechazar
ILUMINACION_MAX    = 220    # Brillo máximo promedio — sobreexpuesto = rechazar
FRONTAL_MARGEN     = 0.35   # Rango amplio para considerar cara frontal (35%–65%)

FRAMES_REQUERIDOS   = 10

# ─────────────────────────────────────────────
#  Umbrales de Identidad — v2 (CORREGIDOS)
# ─────────────────────────────────────────────
# Umbral subido: con 32 puntos se puede exigir mayor similitud sin penalizar
# al usuario legítimo. Reduce falsos positivos (persona distinta aceptada).
SIMILITUD_GATE      = 0.94   # Gate en tiempo real  (antes: 0.92 — insuficiente)
SIMILITUD_GATE_DURO = 0.97   # Validación final     (antes: 0.96 — insuficiente)
MAX_FRAMES_RECHAZO  = 25     # Frames de rechazo antes de contar un fallo

# ─────────────────────────────────────────────
#  Anti-Spoofing (anti-video / anti-foto)
# ─────────────────────────────────────────────
# El reto de vivacidad activo es el principal mecanismo anti-video:
# se sortea justo cuando abre la cámara, un video pre-grabado no puede responderlo.
SPOOF_FRAMES_VENTANA   = 20       # Ventana de frames para analizar movimiento
SPOOF_TEXTURA_MIN_LAP  = 8.0      # Varianza mínima de textura (más tolerante)
# Reto de vivacidad activo: el usuario debe completar UN gesto aleatorio antes
# de pasar al gate de identidad. Tiempo límite en segundos.
SPOOF_RETO_TIMEOUT_SEG = 15       # Tiempo máximo para completar el reto activo

# ─────────────────────────────────────────────
#  UI — Accesibilidad (adultos mayores)
# ─────────────────────────────────────────────
FUENTE_GRANDE    = cv2.FONT_HERSHEY_DUPLEX
FUENTE_NORMAL    = cv2.FONT_HERSHEY_SIMPLEX
TEXTO_SCALE_G    = 0.95   # Escala grande para instrucciones principales
TEXTO_SCALE_P    = 0.60   # Escala pequeña para info técnica

# ─────────────────────────────────────────────
#  Anti-Brute Force
# ─────────────────────────────────────────────
MAX_INTENTOS_LOGIN = 3
COOLDOWN_SEGUNDOS  = 30
_intentos_fallidos = {}

# ─────────────────────────────────────────────
#  Rutas
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDIA_DIR = os.path.join(BASE_DIR, "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Poses de Registro
# ─────────────────────────────────────────────
POSES_REGISTRO = [
    {"id": "blink",   "label": "VIDA (1/2): CIERRE Y ABRA LOS OJOS",   "cant": 0, "cond": ["parpadeo"]},
    {"id": "smile",   "label": "VIDA (2/2): SONRIA AMPLIAMENTE",         "cant": 0, "cond": ["sonrisa"]},
    {"id": "frontal", "label": "POSICION (1/1): MIRA AL FRENTE",         "cant": 5, "cond": []},
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
#  Anti-Brute Force
# ─────────────────────────────────────────────
def check_bloqueo(nombre):
    estado = _intentos_fallidos.get(nombre)
    if not estado:
        return False, 0.0
    ahora = time.time()
    if estado["bloqueado_hasta"] and ahora < estado["bloqueado_hasta"]:
        return True, estado["bloqueado_hasta"] - ahora
    return False, 0.0

def registrar_fallo(nombre):
    if nombre not in _intentos_fallidos:
        _intentos_fallidos[nombre] = {"count": 0, "bloqueado_hasta": 0.0}
    _intentos_fallidos[nombre]["count"] += 1
    count = _intentos_fallidos[nombre]["count"]
    print(f" [SEGURIDAD] Fallo #{count} para '{nombre}'.")
    if count >= MAX_INTENTOS_LOGIN:
        _intentos_fallidos[nombre]["bloqueado_hasta"] = time.time() + COOLDOWN_SEGUNDOS
        _intentos_fallidos[nombre]["count"] = 0
        return True
    return False

def resetear_intentos(nombre):
    if nombre in _intentos_fallidos:
        _intentos_fallidos[nombre] = {"count": 0, "bloqueado_hasta": 0.0}

# ─────────────────────────────────────────────
#  Biometría — Fingerprint MEJORADO (24 puntos)
# ─────────────────────────────────────────────
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def get_face_fingerprint(landmarks):
    """
    32 proporciones faciales (XY) + 8 coordenadas Z de profundidad 3D.
    Total: 40 valores. El componente Z es el clave para distinguir caras
    reales 3D de videos/fotos 2D proyectados en pantalla, ya que MediaPipe
    estima profundidad relativa que no se transfiere fielmente en video.
    SOLO usar cuando la cara es frontal (verificar con es_frontal()).
    """
    p = landmarks
    norm = ((p[468].x - p[473].x)**2 + (p[468].y - p[473].y)**2) ** 0.5
    if norm < 1e-6:
        norm = 1e-6

    key_points = [
        # ── Medidas XY originales (24) ──
        (10, 152),   # Altura cara
        (33, 263),   # Ancho entre ojos
        (61, 291),   # Ancho boca
        (1,  4),     # Largo nariz
        (33, 133),   # Ojo izquierdo ancho
        (263, 362),  # Ojo derecho ancho
        (61, 0),     # Boca-nariz izq
        (291, 0),    # Boca-nariz der
        (10, 1),     # Frente-nariz
        (152, 1),    # Menton-nariz
        (234, 454),  # Ancho total cara
        (10, 234),   # Diagonal frente-mejilla
        (159, 145),  # Apertura ojo izquierdo
        (386, 374),  # Apertura ojo derecho
        (13, 14),    # Apertura labios vertical
        (78, 308),   # Comisuras boca internas
        (70, 300),   # Cejas extremos
        (105, 334),  # Cejas internas
        (9, 151),    # Centro frente a menton
        (127, 356),  # Pomulos
        (93, 323),   # Mandibula lateral
        (152, 377),  # Menton-labio inferior
        (2, 164),    # Punta nariz-labio superior
        (468, 473),  # Distancia interpupilar (ancla)
        # ── Puntos adicionales v2 (+8) — más discriminativos ──
        (55, 285),   # Arco ceja izquierda (pico)
        (107, 336),  # Arco ceja derecha (pico)
        (168, 6),    # Raiz nariz-entrecejo
        (57, 287),   # Angulo comisura-mejilla izq
        (17, 200),   # Labio inferior-menton
        (164, 393),  # Filtrum (surco nasolabial)
        (234, 152),  # Diagonal mejilla-menton izq
        (454, 152),  # Diagonal mejilla-menton der
    ]

    fp = []
    for (i, j) in key_points:
        dist = ((p[i].x - p[j].x)**2 + (p[i].y - p[j].y)**2) ** 0.5
        fp.append(dist / norm)

    # ── Coordenadas Z de profundidad 3D (+8) ──
    # Z está normalizado por MediaPipe relativo a la cara.
    # Usamos la misma norma (distancia interpupilar) que para X e Y, ya que
    # MediaPipe escala Z en la misma proporción general que X, garantizando estabilidad.
    puntos_z = [1, 4, 33, 263, 61, 291, 10, 152]  # nariz, punta, ojos, boca, frente, menton
    for idx in puntos_z:
        fp.append(p[idx].z / norm)

    return fp

def es_frontal(landmarks):
    """
    True si la nariz está en el rango central del ancho de la cara.
    Usa FRONTAL_MARGEN (default 35%-65%) — más amplio que antes (40%-60%)
    para no rechazar caras válidas por micro-movimientos.
    """
    p     = landmarks
    nariz = p[1].x
    izq   = p[234].x
    der   = p[454].x
    ancho = der - izq
    if ancho < 1e-6:
        return False
    relativo = (nariz - izq) / ancho
    return FRONTAL_MARGEN < relativo < (1.0 - FRONTAL_MARGEN)

def evaluar_iluminacion(frame):
    """
    Retorna (ok: bool, mensaje: str, brillo: int).
    Rechaza frames demasiado oscuros o sobreexpuestos antes de capturar.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brillo = int(np.mean(gray))
    if brillo < ILUMINACION_MIN:
        return False, f"MUY OSCURO ({brillo}) — MEJORA LA ILUMINACION", brillo
    if brillo > ILUMINACION_MAX:
        return False, f"SOBREEXPUESTO ({brillo}) — MENOS LUZ DIRECTA", brillo
    return True, f"Iluminacion OK ({brillo})", brillo

def calcular_borrosidad(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ─────────────────────────────────────────────
#  Anti-Spoofing — Reto Activo Aleatorio
# ─────────────────────────────────────────────
# Lista de retos posibles. Se sortea UNO al iniciar cada sesión de login.
# Un video pre-grabado no puede predecir cuál se va a pedir.
RETOS_VIVACIDAD = [
    {"id": "parpadeo",  "label": "PARPADEE AHORA",       "cond": "parpadeo",  "arrow": None},
    {"id": "sonrisa",   "label": "SONRIA AHORA",          "cond": "sonrisa",   "arrow": None},
]

class AntiSpoofingActivo:
    """
    Detección de vivacidad mediante RETO ACTIVO ALEATORIO.

    Estrategia principal: sortear un gesto al iniciar el login y mostrarlo
    en pantalla justo entonces. Un video pre-grabado NO puede prepararse para
    un reto que se decide en tiempo real.

    Secundario: verificar textura de piel en zona nasal (rechaza fotos impresas).
    """
    def __init__(self):
        # Sortear reto aleatorio
        self.reto          = random.choice(RETOS_VIVACIDAD)
        self.superado      = False
        self.frames_reto   = 0          # frames consecutivos que cumplen el reto
        self.inicio_ts     = time.time()
        self.alerta        = ""
        self.blink_estado  = "abierto"
        self.blink_frames  = 0
        print(f" [ANTISPOOF] Reto activo sorteado: '{self.reto['id']}'")

    @property
    def tiempo_restante(self):
        return max(0.0, SPOOF_RETO_TIMEOUT_SEG - (time.time() - self.inicio_ts))

    def analizar(self, frame, face_lm, acciones):
        """
        Retorna (superado: bool, alerta: str).
        superado=True una vez que el usuario completa el reto.
        Después de superado, ya no bloquea.
        """
        if self.superado:
            return True, ""

        pts = face_lm.landmark

        # ── Check secundario: textura de piel (foto impresa muy borrosa) ──
        h, w = frame.shape[:2]
        nx = int(pts[1].x * w)
        ny = int(pts[1].y * h)
        mg = 35
        roi = frame[max(0,ny-mg):min(h,ny+mg), max(0,nx-mg):min(w,nx+mg)]
        if roi.size > 0:
            lap = calcular_borrosidad(roi)
            if lap < SPOOF_TEXTURA_MIN_LAP:
                self.alerta = "IMAGEN PLANA DETECTADA — USE SU CARA REAL"
                return False, self.alerta

        # ── Check principal: reto activo ──
        cond = self.reto["cond"]
        cumple = acciones.get(cond, False)

        if cond == "parpadeo":
            if self.blink_estado == "abierto" and cumple:
                self.blink_estado = "cerrado"
                self.blink_frames = 0
            elif self.blink_estado == "cerrado":
                if cumple:
                    self.blink_frames += 1
                else:
                    if self.blink_frames >= 1:
                        self.superado = True
                        print(f" [ANTISPOOF] Reto 'parpadeo' superado.")
                        return True, ""
                    self.blink_estado = "abierto"
        else:
            if cumple:
                self.frames_reto += 1
            else:
                self.frames_reto = 0

            # 8 frames consecutivos cumpliendo el reto = superado
            if self.frames_reto >= 8:
                self.superado = True
                print(f" [ANTISPOOF] Reto '{self.reto['id']}' superado.")
                return True, ""

        # Timeout: no completó el reto a tiempo
        if self.tiempo_restante <= 0:
            self.alerta = "TIEMPO AGOTADO — REINICIANDO"
            return False, self.alerta

        self.alerta = ""
        return False, ""

    def reset(self):
        self.__init__()

def contar_dedos(hand_landmarks):
    if not hand_landmarks:
        return 0
    tips  = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    dedos = 0
    if hand_landmarks[4].x < hand_landmarks[3].x:
        dedos += 1
    for tip, base in zip(tips, bases):
        if hand_landmarks[tip].y < hand_landmarks[base].y:
            dedos += 1
    return dedos

def evaluar_posicion_mano(pts_face, pts_hand, zona_esperada):
    if not pts_face or not pts_hand:
        return False
    f_izq = pts_face[234].x
    f_der = pts_face[454].x
    f_abj = pts_face[152].y
    m_x   = pts_hand[0].x
    m_y   = pts_hand[0].y
    margen = 0.05
    if zona_esperada == "DERECHA":
        return m_x > f_der + margen
    if zona_esperada == "IZQUIERDA":
        return m_x < f_izq - margen
    if zona_esperada == "ABAJO":
        return m_y > f_abj + margen
    return False

def detectar_acciones(face_landmarks, hand_landmarks=None):
    acciones = {
        "parpadeo": False, "sonrisa": False, "giro_der": False,
        "giro_izq": False, "arriba": False, "abajo": False,
        "dedos": 0, "mano_zona": None
    }
    if not face_landmarks:
        return acciones
    pts = face_landmarks.landmark

    if abs(pts[159].y - pts[145].y) < EAR_UMBRAL:
        acciones["parpadeo"] = True
    if abs(pts[61].x - pts[291].x) > BOCA_UMBRAL:
        acciones["sonrisa"] = True

    nariz = pts[1]
    if nariz.x > pts[454].x - GIRO_DER_UMBRAL:
        acciones["giro_der"] = True
    if nariz.x < pts[234].x + GIRO_IZQ_UMBRAL:
        acciones["giro_izq"] = True

    frente = pts[10]
    menton = pts[152]
    if nariz.y < frente.y + ARRIBA_UMBRAL:
        acciones["arriba"] = True
    if nariz.y > menton.y - (menton.y - frente.y) * ABAJO_FACTOR:
        acciones["abajo"] = True

    if hand_landmarks:
        acciones["dedos"] = contar_dedos(hand_landmarks.landmark)
    return acciones

# ─────────────────────────────────────────────
#  UI helpers — Accesibles para adultos mayores
# ─────────────────────────────────────────────
def put_texto_grande(frame, texto, y, color=(255, 255, 255), escala=None):
    """
    Dibuja texto con sombra para mayor legibilidad.
    Auto-escala el tamaño para que el texto siempre quepa en pantalla.
    """
    h, w = frame.shape[:2]
    margen_x = 40
    ancho_max = w - margen_x * 2

    # Calcular escala máxima que cabe en el ancho disponible
    escala_base = TEXTO_SCALE_G if escala is None else escala
    (tw, _), _ = cv2.getTextSize(texto, FUENTE_GRANDE, escala_base, 2)
    if tw > ancho_max:
        escala_base = escala_base * (ancho_max / tw)

    x = margen_x
    # Sombra negra
    cv2.putText(frame, texto, (x + 2, y + 2), FUENTE_GRANDE, escala_base, (0, 0, 0), 3)
    # Texto principal
    cv2.putText(frame, texto, (x, y), FUENTE_GRANDE, escala_base, color, 2)

def draw_arrow(frame, direction):
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color  = (0, 255, 255)
    length = 100
    thick  = 8
    if direction == "DERECHA":
        cv2.arrowedLine(frame, (cx - length, cy), (cx + length, cy), color, thick)
    elif direction == "IZQUIERDA":
        cv2.arrowedLine(frame, (cx + length, cy), (cx - length, cy), color, thick)
    elif direction == "ARRIBA":
        cv2.arrowedLine(frame, (cx, cy + length), (cx, cy - length), color, thick)
    elif direction == "ABAJO":
        cv2.arrowedLine(frame, (cx, cy - length), (cx, cy + length), color, thick)

def draw_target_zone(frame, zona):
    h, w  = frame.shape[:2]
    color = (255, 255, 0)
    if zona == "DERECHA":
        cv2.rectangle(frame, (w - 200, h//2 - 100), (w - 20, h//2 + 100), color, 2)
        cv2.putText(frame, "MANO AQUI", (w - 180, h//2 - 110), 1, 1, color, 1)
    elif zona == "IZQUIERDA":
        cv2.rectangle(frame, (20, h//2 - 100), (200, h//2 + 100), color, 2)
        cv2.putText(frame, "MANO AQUI", (40, h//2 - 110), 1, 1, color, 1)
    elif zona == "ABAJO":
        cv2.rectangle(frame, (w//2 - 100, h - 180), (w//2 + 100, h - 20), color, 2)
        cv2.putText(frame, "MANO AQUI", (w//2 - 80, h - 190), 1, 1, color, 1)

def cuenta_regresiva(cap, mensaje, segundos=3):
    start = time.time()
    while time.time() - start < segundos:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 10, 5), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        # Texto más grande y con sombra para legibilidad
        secs_rest = str(segundos - int(time.time() - start))
        cv2.putText(frame, mensaje.upper(), (w//2 - 280, h//2 - 60),
                    FUENTE_GRANDE, 1.1, (0, 0, 0), 4)
        cv2.putText(frame, mensaje.upper(), (w//2 - 280, h//2 - 60),
                    FUENTE_GRANDE, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, secs_rest,
                    (w//2 - 35, h//2 + 90), FUENTE_GRANDE, 4.5, (0, 0, 0), 8)
        cv2.putText(frame, secs_rest,
                    (w//2 - 35, h//2 + 90), FUENTE_GRANDE, 4.5, (0, 255, 0), 5)
        cv2.imshow("StarPulse Ultra", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def mostrar_progreso(frame, paso, total, subpaso_prog=0):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (50, h-40), (w-50, h-20), (40, 40, 40), -1)
    ancho_total  = w - 100
    progreso_gen = min((paso + subpaso_prog) / total, 1.0)
    cv2.rectangle(frame, (50, h-40),
                  (50 + int(ancho_total * progreso_gen), h-20), (0, 255, 0), -1)
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

def mostrar_indicador_identidad(frame, sim, gate):
    h, w   = frame.shape[:2]
    x0, y0 = w - 230, 70
    x1, y1 = w - 20,  96
    color  = (0, 220, 0) if sim >= gate else (0, 0, 220)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (50, 50, 50), -1)
    fill = int((x1 - x0) * min(sim, 1.0))
    cv2.rectangle(frame, (x0, y0), (x0 + fill, y1), color, -1)
    gate_x = x0 + int((x1 - x0) * gate)
    cv2.line(frame, (gate_x, y0 - 4), (gate_x, y1 + 4), (255, 255, 0), 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)
    cv2.putText(frame, f"ID: {sim*100:.1f}%  (min {gate*100:.0f}%)",
                (x0, y0 - 8), FUENTE_NORMAL, 0.48, color, 1)

def mostrar_indicador_antispoof(frame, superado, tiempo_rest, reto_label, alerta=""):
    """Indicador visual del reto activo de vivacidad."""
    h, w = frame.shape[:2]
    x0, y0 = w - 230, 105
    x1, y1 = w - 20, 127
    color = (0, 200, 0) if superado else (0, 160, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (50, 50, 50), -1)
    if not superado and tiempo_rest > 0:
        fill = int((x1 - x0) * (tiempo_rest / SPOOF_RETO_TIMEOUT_SEG))
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y1), color, -1)
    elif superado:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)
    label = "VIVO OK" if superado else f"RETO: {reto_label}"
    cv2.putText(frame, label, (x0, y0 - 8), FUENTE_NORMAL, 0.44, color, 1)
    if alerta:
        cv2.putText(frame, alerta, (40, h - 65),
                    FUENTE_NORMAL, 0.6, (0, 100, 255), 2)

# ─────────────────────────────────────────────
#  Flujo biometrico principal
# ─────────────────────────────────────────────
def flujo_biometrico(modo, nombre, fingerprint_esperado=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se encontro camara.")
        return False, None

    # ══════════════════════════════════════════
    #  MODO REGISTRO
    # ══════════════════════════════════════════
    if modo == "REGISTRO":
        paso_actual            = 0
        pool_poses_capturadas  = {}
        fingerprint_referencia = None
        fps_frontales          = []   # Solo acumular FPs de frames frontales
        mejor_fingerprint      = None
        frames_pose            = 0
        capturas_pose_actual   = 0
        mejor_frame_pose       = None
        mejor_lap_pose         = -1
        # Detección de ciclos de parpadeo (abre → cierra → abre = 1 ciclo)
        blink_estado           = "abierto"  # "abierto" | "cerrado"
        blink_ciclos           = 0
        blink_frames_cerrado   = 0

        cuenta_regresiva(cap, "INICIANDO REGISTRO GUIADO")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_face = face_mesh.process(rgb)
            res_hands= hands.process(rgb)

            cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)
            cv2.putText(frame, f"StarPulse  |  REGISTRO  |  {nombre}", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            if res_face.multi_face_landmarks:
                face_lm    = res_face.multi_face_landmarks[0]
                hand_lm    = (res_hands.multi_hand_landmarks[0]
                              if res_hands.multi_hand_landmarks else None)
                acciones   = detectar_acciones(face_lm, hand_lm)
                current_fp = get_face_fingerprint(face_lm.landmark)

                # Nota: NO se verifica consistencia de identidad durante registro.
                # Las expresiones de vida (parpadeo, sonrisa) cambian proporciones.
                # La guarda definitiva es el check anti-duplicado al final.

                if paso_actual < len(POSES_REGISTRO):
                    pose       = POSES_REGISTRO[paso_actual]
                    cumple_pose= all(acciones[c] for c in pose["cond"])

                    if pose["id"] == "blink":
                        # Detectar CICLO de parpadeo independientemente de cumple_pose
                        ojo_cerrado = acciones["parpadeo"]
                        if blink_estado == "abierto" and ojo_cerrado:
                            blink_estado = "cerrado"
                            blink_frames_cerrado = 0
                        elif blink_estado == "cerrado":
                            if ojo_cerrado:
                                blink_frames_cerrado += 1
                            else:
                                # Ojo abrió de nuevo → ciclo completo
                                if blink_frames_cerrado >= 1:
                                    blink_ciclos += 1
                                    print(f" Parpadeo #{blink_ciclos} ({blink_frames_cerrado} frames)")
                                blink_estado = "abierto"

                        # Mostrar estado textual y hacer que barra se llene si se completó el ciclo
                        put_texto_grande(frame,
                            f"Parpadeos detectados: {blink_ciclos}/1",
                            190, color=(0, 255, 180), escala=0.65)
                        
                        frames_pose = 20 if blink_ciclos >= 1 else 0

                        if blink_ciclos >= 1:
                            print(f" OK Vida 'blink' (1 ciclo).")
                            if not fingerprint_referencia:
                                if es_frontal(face_lm.landmark):
                                    fingerprint_referencia = current_fp
                                    print(" Identidad anclada (frontal).")
                                else:
                                    # Espera a estar de frente
                                    pass
                            
                            if fingerprint_referencia:
                                paso_actual  += 1
                                frames_pose   = 0
                                blink_ciclos  = 0
                                blink_estado  = "abierto"
                                mensaje_camara(frame, "VIDA VERIFICADA", (0, 150, 0), 1000)

                    else:
                        # Otros retos usan cumple_pose
                        if cumple_pose:
                            frames_pose += 1

                            if pose["cant"] == 0:   # Reto de vida sostenido (sonrisa, giro)
                                if frames_pose >= 20:
                                    print(f" OK Vida '{pose['id']}'.")
                                    if not fingerprint_referencia:
                                        if es_frontal(face_lm.landmark):
                                            fingerprint_referencia = current_fp
                                            print(" Identidad anclada (frontal).")
                                        else:
                                            frames_pose = 0
                                            continue
                                    paso_actual += 1
                                    frames_pose  = 0
                                    mensaje_camara(frame, "VIDA VERIFICADA", (0, 150, 0), 1000)

                            else:   # Captura de pose
                                if frames_pose >= 10:
                                    if capturas_pose_actual < 5:
                                        # —— Verificar que la cara está en la posición correcta ——
                                        posicion_ok = True
                                        if pose["id"] == "frontal":
                                            posicion_ok = es_frontal(face_lm.landmark)
                                            if not posicion_ok:
                                                put_texto_grande(frame, "MIRA DE FRENTE A LA CAMARA",
                                                                 190, color=(0, 200, 255), escala=0.65)
                                        # Para otras poses, la condicion ya fue verificada en cumple_pose

                                        if not posicion_ok:
                                            cv2.imshow("StarPulse Ultra", frame)
                                            cv2.waitKey(1)
                                            continue

                                        # Validar iluminacion antes de capturar
                                        luz_ok, luz_msg, brillo = evaluar_iluminacion(frame)
                                        if not luz_ok:
                                            put_texto_grande(frame, luz_msg, 190,
                                                             color=(0, 140, 255), escala=0.65)
                                            cv2.imshow("StarPulse Ultra", frame)
                                            cv2.waitKey(1)
                                            continue

                                        lap = calcular_borrosidad(frame)
                                        if lap > mejor_lap_pose:
                                            mejor_lap_pose  = lap
                                            mejor_frame_pose= frame.copy()

                                        if pose["id"] == "frontal":
                                            # es_frontal() ya verificado arriba
                                            fps_frontales.append(current_fp)
                                            print(f" FP frontal #{len(fps_frontales)} (Lap={lap:.0f}, Luz={brillo})")

                                        capturas_pose_actual += 1
                                        mensaje_camara(frame,
                                                       f"CAPTURA {capturas_pose_actual}/5 ({pose['id']})",
                                                       (200, 100, 0), 800)
                                    else:
                                        pool_poses_capturadas[pose["id"]] = mejor_frame_pose
                                        print(f" OK Pose '{pose['id']}'.")
                                        paso_actual         += 1
                                        frames_pose          = 0
                                        capturas_pose_actual = 0
                                        mejor_frame_pose     = None
                                        mejor_lap_pose       = -1
                                        if paso_actual < len(POSES_REGISTRO):
                                            mensaje_camara(frame, "POSICION OK", (0, 150, 0), 800)
                        else:
                            frames_pose = 0

                    if paso_actual < len(POSES_REGISTRO):
                        put_texto_grande(frame, POSES_REGISTRO[paso_actual]["label"],
                                         110, color=(0, 255, 255))
                        if "arrow" in POSES_REGISTRO[paso_actual]:
                            draw_arrow(frame, POSES_REGISTRO[paso_actual]["arrow"])

                    prog = ((capturas_pose_actual / 5.0) if pose["cant"] > 0
                            else (frames_pose / 20.0))
                    mostrar_progreso(frame, paso_actual, len(POSES_REGISTRO),
                                     prog * (1.0 / len(POSES_REGISTRO)))

                # Finalizar registro
                if paso_actual == len(POSES_REGISTRO):
                    if not fps_frontales:
                        # Fallback: usar el fingerprint de referencia (tomado al anclar identidad)
                        if fingerprint_referencia:
                            print(" AVISO: 0 FPs frontales capturados durante poses.")
                            print(" Usando fingerprint_referencia como fallback.")
                            fps_frontales = [fingerprint_referencia]
                        else:
                            print(" ERROR: Sin FPs frontales y sin referencia.")
                            mensaje_camara(frame,
                                           "ERROR: ILUMINACION INSUFICIENTE — REGISTRESE CON MAS LUZ",
                                           (0, 0, 200), 4000)
                            cap.release()
                            cv2.destroyAllWindows()
                            return False, None

                    # Promediar → vector mas robusto y estable
                    mejor_fingerprint = list(np.mean(fps_frontales, axis=0))
                    print(f" FP final: promedio de {len(fps_frontales)} frames frontales.")
                    print(f" Valores muestra: {[round(x,4) for x in mejor_fingerprint[:6]]}")

                    print(" Verificando unicidad...")
                    res_dup = existe_cara_en_db(mejor_fingerprint)
                    if res_dup["existe"]:
                        mensaje_camara(frame, f"CARA YA REGISTRADA ({res_dup['nombre']})",
                                       (0, 0, 200), 4000)
                        print(f" DUPLICADO: '{res_dup['nombre']}'.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False, None

                    for pid, img in pool_poses_capturadas.items():
                        path = os.path.join(MEDIA_DIR, f"reg_{nombre}_{pid}.jpg")
                        cv2.imwrite(path, img)
                        print(f" Guardado: {path}")

                    mensaje_camara(frame, "REGISTRO COMPLETADO", (0, 200, 0), 2000)
                    cap.release()
                    cv2.destroyAllWindows()
                    return True, mejor_fingerprint

            else:
                cv2.putText(frame, "NO SE DETECTA ROSTRO", (50, 110),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("StarPulse Ultra", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return False, None

    # ══════════════════════════════════════════
    #  MODO LOGIN
    # ══════════════════════════════════════════
    else:
        # Retos de seguridad: SOLO gestos faciales aleatorios.
        # Sin manos — evita confusión y ataques donde otra persona pone la mano.
        POOL_RETOS = [
            {"label": "PARPADEE",               "cond": ["parpadeo"],            "arrow": None},
            {"label": "SONRIA AMPLIO",           "cond": ["sonrisa"],             "arrow": None},
            {"label": "SONRIA Y PARPADEE",      "cond": ["sonrisa", "parpadeo"], "arrow": None},
        ]

        retos_actuales   = random.sample(POOL_RETOS, 2)
        paso_reto        = 0
        frames_cumplidos = 0
        frames_fail_id   = 0
        exito_final      = False
        fp_ref           = np.array(fingerprint_esperado, dtype=float)

        # Ultimo sim calculado (para no recalcular si no es frontal)
        ultima_sim = 0.0

        # Estado para seguimiento de parpadeos en los retos de seguridad
        blink_estado_seg = "abierto"
        blink_frames_seg = 0

        # Reto activo de vivacidad — sorteado aquí, justo antes de abrir cámara

        anti_spoof = AntiSpoofingActivo()
        print(f" [LOGIN] Reto vivacidad: {anti_spoof.reto['label']}")

        cuenta_regresiva(cap, "INICIANDO VERIFICACION BIOMETRICA")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_face = face_mesh.process(rgb)
            res_hands= hands.process(rgb)

            cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)
            cv2.putText(frame, f"StarPulse  |  LOGIN  |  {nombre}", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            if res_face.multi_face_landmarks:
                face_lm  = res_face.multi_face_landmarks[0]
                acciones = detectar_acciones(face_lm)   # sin manos

                # ══════════════════════════
                #  RETO ACTIVO DE VIVACIDAD (PRIMERO)
                # ══════════════════════════
                spoof_ok, spoof_alerta = anti_spoof.analizar(frame, face_lm, acciones)
                mostrar_indicador_antispoof(
                    frame, spoof_ok,
                    anti_spoof.tiempo_restante,
                    anti_spoof.reto["label"],
                    spoof_alerta
                )

                if not spoof_ok:
                    # Mostrar el reto en pantalla de forma prominente
                    put_texto_grande(frame, "PRUEBA DE VIVACIDAD:", 110, color=(255, 200, 0))
                    put_texto_grande(frame, anti_spoof.reto["label"],
                                     155, color=(0, 255, 255))
                    if anti_spoof.reto["arrow"]:
                        draw_arrow(frame, anti_spoof.reto["arrow"])

                    # Timeout: reiniciar con reto nuevo
                    if anti_spoof.tiempo_restante <= 0:
                        put_texto_grande(frame, "TIEMPO AGOTADO — INTENTE DE NUEVO",
                                         200, color=(0, 50, 255))
                        cv2.imshow("StarPulse Ultra", frame)
                        cv2.waitKey(2000)
                        anti_spoof.reset()  # sortea nuevo reto

                    cv2.imshow("StarPulse Ultra", frame)
                    cv2.waitKey(1)
                    continue  # no pasa al gate hasta superar el reto

                # ══════════════════════════
                #  GATE DE IDENTIDAD
                # ══════════════════════════
                frontal    = es_frontal(face_lm.landmark)
                current_fp = get_face_fingerprint(face_lm.landmark)

                # Validar iluminacion — si es mala, pausar y avisar sin penalizar
                luz_ok, luz_msg, brillo = evaluar_iluminacion(frame)
                if not luz_ok:
                    put_texto_grande(frame, luz_msg, 110, color=(0, 140, 255))
                    put_texto_grande(frame, "AJUSTE LA ILUMINACION PARA CONTINUAR",
                                     155, color=(0, 140, 255), escala=0.60)
                    cv2.imshow("StarPulse Ultra", frame)
                    cv2.waitKey(1)
                    continue  # No penaliza, solo espera

                if frontal:
                    # Solo calcular similitud cuando la cara esta de frente
                    ultima_sim = cosine_similarity(current_fp, fp_ref)
                    print(f" [DEBUG] sim={ultima_sim*100:.2f}%  frontal=True  fail={frames_fail_id}")

                sim = ultima_sim
                mostrar_indicador_identidad(frame, sim, SIMILITUD_GATE)

                # ── FIREWALL DE IDENTIDAD ──
                if frontal and sim < SIMILITUD_GATE:
                    frames_fail_id  += 1
                    frames_cumplidos = 0

                    put_texto_grande(frame, "ROSTRO NO RECONOCIDO", 110, color=(0, 0, 255))
                    put_texto_grande(frame,
                        f"Coincidencia: {sim*100:.1f}%  (necesita {SIMILITUD_GATE*100:.0f}%)",
                        155, color=(80, 80, 255), escala=0.60)

                    if frames_fail_id >= MAX_FRAMES_RECHAZO:
                        bloqueado = registrar_fallo(nombre)
                        if bloqueado:
                            msg = f"BLOQUEADO {COOLDOWN_SEGUNDOS}s — DEMASIADOS INTENTOS"
                            print(f" [SEGURIDAD] {nombre} bloqueado.")
                        else:
                            cnt = _intentos_fallidos.get(nombre, {}).get("count", 0)
                            msg = f"ACCESO DENEGADO"
                            print(f" [SEGURIDAD] Gate fallido. Restantes: {MAX_INTENTOS_LOGIN - cnt}")
                        mensaje_camara(frame, msg, (0, 0, 180), 3500)
                        break

                    cv2.imshow("StarPulse Ultra", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue  # NO avanza a los retos

                # Identidad valida
                if frontal and sim >= SIMILITUD_GATE:
                    frames_fail_id = 0

                # ══════════════════════════
                #  RETOS DE SEGURIDAD (solo cara)
                # ══════════════════════════
                if paso_reto < len(retos_actuales):
                    reto = retos_actuales[paso_reto]

                    put_texto_grande(frame, f"PASO {paso_reto+1}/{len(retos_actuales)}:",
                                     110, color=(255, 200, 0))
                    put_texto_grande(frame, reto["label"], 155, color=(0, 255, 0))
                    if reto["arrow"]:
                        draw_arrow(frame, reto["arrow"])

                    # Verificar que la identidad no decaiga durante el reto
                    if frontal and sim < SIMILITUD_GATE:
                        frames_cumplidos = 0
                        mensaje_camara(frame, "IDENTIDAD INCONSISTENTE — REINICIANDO",
                                       (0, 0, 200), 1200)
                    else:
                        # Si el reto requiere parpadear, usar detector de ciclo
                        requiere_parpadeo = ("parpadeo" in reto["cond"])
                        otras_condiciones = [c for c in reto["cond"] if c != "parpadeo"]

                        cumple_otras = all(acciones.get(c, False) for c in otras_condiciones)

                        if requiere_parpadeo:
                            # Requiere parpadear (y posiblemente otras cosas, como sonreir)
                            if cumple_otras:
                                ojo_cerrado = acciones.get("parpadeo", False)
                                if blink_estado_seg == "abierto" and ojo_cerrado:
                                    blink_estado_seg = "cerrado"
                                    blink_frames_seg = 0
                                elif blink_estado_seg == "cerrado":
                                    if ojo_cerrado:
                                        blink_frames_seg += 1
                                    else:
                                        if blink_frames_seg >= 1:
                                            frames_cumplidos = FRAMES_REQUERIDOS # Fuerza el avance
                                        blink_estado_seg = "abierto"
                            else:
                                blink_estado_seg = "abierto"
                                frames_cumplidos = 0
                        else:
                            # Reto normal sin parpadeo (sonreir, girar, etc) requiere frames sostenidos
                            if cumple_otras:
                                frames_cumplidos += 1
                            else:
                                frames_cumplidos = 0

                    if frames_cumplidos >= FRAMES_REQUERIDOS:
                        paso_reto       += 1
                        frames_cumplidos = 0
                        blink_estado_seg = "abierto"

                        if paso_reto < len(retos_actuales):
                            mensaje_camara(frame, "MUY BIEN — SIGUIENTE", (0, 150, 0), 800)
                        else:
                            # Validacion final estricta
                            sim_final = cosine_similarity(
                                get_face_fingerprint(face_lm.landmark), fp_ref)
                            print(f" [FINAL] sim={sim_final*100:.2f}%  umbral={SIMILITUD_GATE_DURO*100:.0f}%")

                            if sim_final >= SIMILITUD_GATE_DURO:
                                resetear_intentos(nombre)
                                print(" ACCESO CONCEDIDO.")
                                mensaje_camara(frame, "BIENVENIDO — ACCESO CONCEDIDO",
                                               (0, 200, 0), 2500)
                                exito_final = True
                            else:
                                bloqueado = registrar_fallo(nombre)
                                msg = ("DEMASIADOS INTENTOS — ESPERE"
                                       if bloqueado
                                       else "VERIFICACION FALLIDA")
                                print(f" DENEGADO: sim final {sim_final*100:.2f}%")
                                mensaje_camara(frame, msg, (0, 0, 200), 3000)
                                exito_final = False
                            break

                    prog = min(frames_cumplidos / float(FRAMES_REQUERIDOS), 1.0)
                    mostrar_progreso(frame, paso_reto, len(retos_actuales), prog)

            else:
                put_texto_grande(frame, "NO SE DETECTA SU ROSTRO", 110, color=(0, 0, 255))
                put_texto_grande(frame, "ACERQUESE Y MIRE A LA CAMARA",
                                 155, color=(80, 80, 255), escala=0.65)

            cv2.imshow("StarPulse Ultra", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return exito_final, None

# ─────────────────────────────────────────────
#  Menu principal
# ─────────────────────────────────────────────
def main():
    while True:
        print("\n╔══════════════════════════════════╗")
        print("║      StarPulse Ultra  -  ZK     ║")
        print("╠══════════════════════════════════╣")
        print("║  [1] Registrarse                ║")
        print("║  [2] Iniciar sesion             ║")
        print("║  [3] Salir                      ║")
        print("╚══════════════════════════════════╝")
        op = input("\nSeleccione: ").strip()

        if op == "1":
            nombre = input("Nombre de usuario: ").strip()
            if not nombre:
                print("El nombre no puede estar vacio.")
                continue
            if nombre_existe(nombre):
                print(f"El usuario '{nombre}' ya existe.")
                continue

            password = getpass.getpass("Contrasena: ")
            confirm  = getpass.getpass("Confirmar contrasena: ")
            if password != confirm:
                print("Las contrasenas no coinciden.")
                continue
            if len(password) < 4:
                print("La contrasena debe tener al menos 4 caracteres.")
                continue

            input("\nPresione [ENTER] para abrir la camara...")
            exito, fingerprint = flujo_biometrico("REGISTRO", nombre)

            if exito:
                if registrar_en_servidor(nombre, password, fingerprint):
                    print(f"Usuario '{nombre}' registrado.")
                else:
                    print("Error al guardar en el servidor.")
            else:
                print("Registro cancelado o fallido.")

        elif op == "2":
            nombre   = input("Usuario: ").strip()
            password = getpass.getpass("Contrasena: ")

            bloqueado, segs = check_bloqueo(nombre)
            if bloqueado:
                print(f"Bloqueado. Espere {int(segs)}s.")
                continue

            print("\nVerificando credenciales...")
            res_creds = verificar_credenciales(nombre, password)
            if not res_creds:
                bloqueado = registrar_fallo(nombre)
                if bloqueado:
                    print(f"Bloqueado {COOLDOWN_SEGUNDOS}s por demasiados intentos.")
                else:
                    cnt = _intentos_fallidos.get(nombre, {}).get("count", 0)
                    print(f"Credenciales incorrectas. Intentos restantes: {MAX_INTENTOS_LOGIN - cnt}")
                continue

            print(f"Credenciales OK. Iniciando verificacion biometrica...")
            input("Presione [ENTER] para la camara...")
            exito, _ = flujo_biometrico("LOGIN", nombre,
                                        fingerprint_esperado=res_creds.get("fingerprint"))
            if exito:
                print(f"\nBIENVENIDO, {nombre}. ACCESO CONCEDIDO.")
            else:
                print("Verificacion biometrica fallida.")

        elif op == "3":
            print("Saliendo...")
            break
        else:
            print("Opcion no valida.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSaliendo... Hasta luego.")