from flask import Flask, jsonify, request
import json
import os
import hashlib

import numpy as np

app = Flask(__name__)
DB_FILE = 'database.json'

# ─────────────────────────────────────────────
#  Helpers de base de datos
# ─────────────────────────────────────────────
def cargar_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as f:
            json.dump([], f)
    with open(DB_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def guardar_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def hashear(password: str) -> str:
    """SHA-256 simple. Para producción usar bcrypt."""
    return hashlib.sha256(password.encode()).hexdigest()

def calcular_similitud(fp1, fp2):
    """Calcula la similitud de coseno entre dos huellas faciales."""
    if not fp1 or not fp2: return 0.0
    v1, v2 = np.array(fp1), np.array(fp2)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm == 0: return 0.0
    return np.dot(v1, v2) / norm

# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────
@app.route('/usuarios', methods=['GET'])
def listar():
    db = cargar_db()
    # No exponer el hash de contraseña ni datos biometricos sensibles
    seguros = [{"id": u["id"], "nombre": u["nombre"]} for u in db]
    return jsonify(seguros)


@app.route('/registrar', methods=['POST'])
def registrar():
    """
    Body esperado: { "nombre": str, "password": str, "fingerprint": list }
    Valida:
      - Nombre no vacío
      - Nombre único (case-insensitive)
      - Fingerprint único (similitud)
    """
    datos = request.json or {}
    nombre = (datos.get("nombre") or "").strip()
    password = datos.get("password") or ""
    fingerprint = datos.get("fingerprint") or []

    if not nombre or not password:
        return jsonify({"status": "error", "message": "Datos incompletos"}), 400

    db = cargar_db()

    # 1. Verificar nombre duplicado
    if any(u["nombre"].lower() == nombre.lower() for u in db):
        return jsonify({"status": "error", "message": f"El usuario '{nombre}' ya existe"}), 409

    # 2. Verificar duplicado facial (Anti-duplicidad)
    for u in db:
        if "fingerprint" in u:
            similitud = calcular_similitud(fingerprint, u["fingerprint"])
            if similitud > 0.95: # Umbral de coincidencia alta
                return jsonify({
                    "status": "error", 
                    "message": f"Esta cara ya está registrada como '{u['nombre']}'"
                }), 409

    nuevo_id = str(max((int(u["id"]) for u in db), default=0) + 1)
    nuevo = {
        "id": nuevo_id,
        "nombre": nombre,
        "password_hash": hashear(password),
        "fingerprint": fingerprint
    }
    db.append(nuevo)
    guardar_db(db)
    return jsonify({"status": "success", "id": nuevo_id})


@app.route('/verificar_creds', methods=['POST'])
def verificar_creds():
    datos = request.json or {}
    nombre = (datos.get("nombre") or "").strip()
    password = datos.get("password") or ""

    db = cargar_db()
    usuario = next((u for u in db if u["nombre"].lower() == nombre.lower()), None)

    if not usuario:
        return jsonify({"status": "error", "message": "Usuario no encontrado"}), 401
    if usuario.get("password_hash") != hashear(password):
        return jsonify({"status": "error", "message": "Contraseña incorrecta"}), 401

    return jsonify({
        "status": "success", 
        "nombre": usuario["nombre"],
        "id": usuario["id"],
        "fingerprint": usuario.get("fingerprint")
    })


@app.route('/existe_usuario/<nombre>', methods=['GET'])
def existe_usuario(nombre):
    db = cargar_db()
    existe = any(u["nombre"].lower() == nombre.lower() for u in db)
    return jsonify({"existe": existe})


@app.route('/existe_cara', methods=['POST'])
def existe_cara():
    """Comprueba si una huella facial ya está en la DB."""
    datos = request.json or {}
    fingerprint = datos.get("fingerprint") or []
    db = cargar_db()
    
    for u in db:
        if "fingerprint" in u:
            if calcular_similitud(fingerprint, u["fingerprint"]) > 0.95:
                return jsonify({"existe": True, "nombre": u["nombre"]})
    
    return jsonify({"existe": False})


@app.route('/eliminar/<id>', methods=['DELETE'])
def eliminar(id):
    db = [u for u in cargar_db() if u['id'] != id]
    guardar_db(db)
    return jsonify({"status": "deleted"})


if __name__ == '__main__':
    app.run(port=5000, debug=False)