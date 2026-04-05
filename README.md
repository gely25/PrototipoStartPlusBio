# StarPulse Ultra - Sistema de Autenticación Biométrica (Prototipo CISO)

Este proyecto es una prueba de concepto (PoC) avanzada de autenticación facial y detección de vivacidad (*liveness detection*), diseñada para mitigar ciberataques comunes de inyección de video, *spoofing* con fotos y *deepfakes* en tiempo real. 

Utiliza **OpenCV** para la gestión de cámara/escena y **MediaPipe** (Face Mesh & Hands) para el rastreo y cálculo geométrico de vectores faciales en 3D.

## Estructura del Proyecto

*   **`src/client/main.py`**: Cliente de escritorio interactivo. Maneja la cámara en tiempo real, interfaz de usuario guiada, extracción de vectores geométricos (fingerprints) y rutinas activas de *anti-spoofing*.
*   **`src/server/app.py`**: Servidor local (API) encargado de cotejar los registros de identidad, verificar credenciales y evitar duplicados.
*   **`src/server/data.json` / `database.json`**: Almacenamiento local persistente donde se guardan los perfiles biométricos.
*   **`media/`**: Directorio donde se guardan las capturas de auditoría de los registros.

## Algoritmos de Anti-Spoofing (Prevención de Fraude)

El núcleo de seguridad de StarPulse Ultra radica en sus múltiples capas de verificación contra *bypass* biométricos:

### 1. Reto de Oclusión Aleatoria (Anti Deepfakes)
Al iniciar un intento de acceso, el usuario es desafiado aleatoriamente a tocar un punto específico de su rostro (Ej. la frente, la nariz, mejilla izquierda o derecha) usando su dedo índice.
*   **¿Qué previene?**: Cuando un *deepfake* o un filtro de remplazo de rostro se enfrenta a una oclusión física (un dedo), los algoritmos de máscara colapsan debido a la falta visual, revelando la farsa. Además, es matemáticamente imposible superar este reto con un video pregrabado tradicional debido a su naturaleza aleatoria generada en cada inicio de sesión.
*   **Implementación**: Geometría Euclidiana midiendo la proximidad de la punta del dedo índice (`Hand Landmark 8`) contra la meta facial en tiempo real.

### 2. Sistema de "Strikes", Timing Dinámico y Validación Estricta (Anti-Video)
En lugar de ser un sistema pasivo que espera infinitamente la "pose correcta" (lo cual permitiría a un atacante inyectar un video de 10 minutos generando movimientos aleatorios hasta atinar casualmente las secuencias), la cámara asume un enfoque penalizador apoyado por una ventana de tiempo y tolerancia ultrarrígida:
*   **Temporizador Visual de Descarte**: Cada desafío de vivacidad (Ej. "Gire la Cabeza") tiene infundido un límite letal de *5 Segundos*, mostrado en pantalla con una barra de desgaste predictiva. Si no se acata la directriz en esa ventana, la inactividad detona un *Strike*.
*   **Validación Negativa (Hiper-Sensibilidad):** Si al usuario se le ordena "Sonreír", y el algoritmo capta que, por el contrario, parpadeó o movió su rostro por apenas 3 fotogramas continuos (es decir, una minúscula décima de segundo extra), se registra como malversación de la posición y arroja un *Strike* inminente. Microgestos erróneos son tratados como indicios de manipulación *random* de software.
*   **Margen Humano y Saneamiento de Buffer**: Para no castigar la motricidad legítima, el sistema amortigua esta hipersensibilidad con un blindaje biológico de *1 Segundo de gracia*. Tras un inicio o advertencia, la cámara suspende las validaciones negativas por instantes mientras activa rutinas de descarte (`cap.grab()`) para evitar evaluar fotogramas viejos encolados y permitir la reanudación al estado neutral sin encadenar castigos múltiples involuntariamente.

### 3. Detección Proactiva de Interferencia
El sistema interrumpe cualquier proceso y lanza una advertencia de bloqueo si capta intentos activos y prematuros de censura de la cámara (identificado internamente como 3 dedos detectados cubriendo el lente de manera indebida).

### 4. Vectorización Geométrica con Eje Z (Profundidad Real 3D)
El *fingerprint* del usuario no solo compara ratios `X` e `Y` de sus facciones, sino que integra de manera algorítmica puntos en el eje `Z`. 
*   **¿Por qué importa?**: Las fotografías impresas y los videos de pantallas son bidimensionales (planos). Al incorporar validación posicional Z extraída desde las mallas 3D, se rechaza la estaticidad plana, pidiendo volumen tridimensional real para conceder el pase final.

### 5. Reconocimiento Activo Continuo (EAR Matemático)
Se usa un cálculo estricto del *Eye Aspect Ratio* (EAR) derivado directamente de la distancia euclidiana entre las comisuras interiores/exteriores y los párpados de ambos ojos para captar parpadeos humanos reales y no solo "formas de ojos cerrados".

## Controles Contra Ataques de Fuerza Bruta
Se incluyó un muro silencioso de bloqueo persistente blindado y centralizado. Todas las fases de la sesión comparten un contador maestro de **3 Vidas o Intentos**.
Acumular 3 faltas por cualquiera de los medios (lentitud térmica, simulación inferior al umbral 99% requerido, o saltarse métricas de gestos dictaminados) cerrará inmediatamente el puerto de escaneo de cámara y condenará la instancia del usuario a un receso nulo de `COOLDOWN_SEGUNDOS` (Mín. 30s), previniendo contundentemente asaltos cibernéticos veloces o suplantadores iterativos.
