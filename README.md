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

### 2. Sistema de "Strikes" y Validación Negativa (Anti-Video)
En lugar de ser un sistema pasivo que espera infinitamente la "pose correcta" (lo cual permitiría a un atacante inyectar un video de 10 minutos de una persona girando hasta que "acierte"), el sistema penaliza con **Strikes**.
*   **Lógica**: Si se le solicita al usuario mantenerse neutral y el sistema detecta que está sonriendo o girando la cabeza, acumulará faltas. Al alcanzar un límite (15 *frames* de error), el sistema deniega el acceso y bloquea la instancia asumiendo un ataque de inyección.

### 3. Detección Proactiva de Interferencia
El sistema interrumpe cualquier proceso y lanza una advertencia de bloqueo si capta intentos activos y prematuros de censura de la cámara (identificado internamente como 3 dedos detectados cubriendo el lente de manera indebida).

### 4. Vectorización Geométrica con Eje Z (Profundidad Real 3D)
El *fingerprint* del usuario no solo compara ratios `X` e `Y` de sus facciones, sino que integra de manera algorítmica puntos en el eje `Z`. 
*   **¿Por qué importa?**: Las fotografías impresas y los videos de pantallas son bidimensionales (planos). Al incorporar validación posicional Z extraída desde las mallas 3D, se rechaza la estaticidad plana, pidiendo volumen tridimensional real para conceder el pase final.

### 5. Reconocimiento Activo Continuo (EAR Matemático)
Se usa un cálculo estricto del *Eye Aspect Ratio* (EAR) derivado directamente de la distancia euclidiana entre las comisuras interiores/exteriores y los párpados de ambos ojos para captar parpadeos humanos reales y no solo "formas de ojos cerrados".

## Controles Contra Ataques de Fuerza Bruta
Se incluyó un sistema silencioso de bloqueo persistente. Múltiples fallos consecutivos de similaridad o de vivacidad bloquearán todo acceso para ese usuario (`COOLDOWN_SEGUNDOS`), forzando retrasos logísticos a cualquier bot de inyecciones iterativas.
