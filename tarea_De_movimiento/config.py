"""
Configuración centralizada para el sistema HAR
"""
import numpy as np

# ============================================
# Configuración de comunicación serial
# ============================================
SERIAL_PORT = 'COM3'  # Cambiar por el puerto de tu Arduino/ESP32
BAUD_RATE = 115200
TIMEOUT = 1

# ============================================
# Parámetros de datos
# ============================================
SENSOR_NUM = 6  # ax, ay, az, gx, gy, gz
STEP_SIZE = 20  # Tamaño de ventana deslizante (timesteps)
SAMPLING_RATE = 50  # Hz

# ============================================
# Clases de actividades (ACTUALIZADO CON RUNNING)
# ============================================
LABEL_DICT = {
    'WAL': 0,   # Walking (Caminando)
    'RUN': 1,   # Running (Corriendo)  ← NUEVA CLASE
    'JUM': 2,   # Jumping (Saltando)
    'FALL': 3,  # Falling (Cayendo)
    'LYI': 4    # Lying (Acostado)
}

CLASS_NAMES = {
    0: 'WAL',
    1: 'RUN',   # ← NUEVA CLASE
    2: 'JUM',
    3: 'FALL',
    4: 'LYI'
}

FULL_NAMES = {
    0: 'Walking',
    1: 'Running',  # ← NUEVA CLASE
    2: 'Jumping',
    3: 'Falling',
    4: 'Lying'
}

NUM_CLASSES = 5  # Actualizado de 4 a 5

# ============================================
# Rutas del modelo y datos
# ============================================
MODEL_PATH = './model_har/'
DATASET_PATH = './dataset/'

# Archivos del modelo
MODEL_FILE = MODEL_PATH + 'model.pkl'
HISTORY_FILE = MODEL_PATH + 'training_history.png'

# ============================================
# Configuración de generación de datos sintéticos
# ============================================
# Duraciones para cada actividad (segundos)
WALKING_DURATION = 5
RUNNING_DURATION = 3    # ← NUEVA CONFIGURACIÓN
JUMPING_DURATION = 2
FALLING_DURATION = 0.5
LYING_DURATION = 3

# Parámetros físicos de movimiento para cada actividad
ACTIVITY_PARAMS = {
    'WAL': {
        'step_freq': 2.0,          # Hz - Frecuencia de pasos al caminar
        'acc_amplitude': 0.5,      # g - Amplitud de aceleración
        'gyro_amplitude': 10       # deg/s - Amplitud de rotación
    },
    'RUN': {                       # ← NUEVOS PARÁMETROS
        'step_freq': 3.5,          # Hz - Frecuencia de pasos al correr (mayor que caminar)
        'acc_amplitude': 1.2,      # g - Amplitud de aceleración (mayor que caminar)
        'gyro_amplitude': 25       # deg/s - Amplitud de rotación (mayor que caminar)
    },
    'JUM': {
        'jump_freq': 1.5,          # Hz - Frecuencia de saltos
        'acc_amplitude': 2.0,      # g - Amplitud de aceleración
        'gyro_amplitude': 20       # deg/s - Amplitud de rotación
    },
    'FALL': {
        'duration': 0.5,           # s - Duración de la caída
        'max_acc': 9.8,           # g - Aceleración máxima
        'max_gyro': 100           # deg/s - Rotación máxima
    },
    'LYI': {
        'noise_level': 0.05       # g - Nivel de ruido cuando está acostado
    }
}

# ============================================
# Configuración de entrenamiento
# ============================================
# Arquitectura del modelo MLP
HIDDEN_LAYERS = (128, 128)  # Dos capas de 128 neuronas cada una
ACTIVATION = 'relu'
LEARNING_RATE = 0.001
MAX_ITER = 100
BATCH_SIZE = 32

# División de datos
TEST_SIZE = 0.3
VALIDATION_FRACTION = 0.1
RANDOM_STATE = 42

# Regularización
ALPHA = 0.001  # L2 regularization parameter

# ============================================
# Configuración de predicción en tiempo real
# ============================================
SMOOTHING_WINDOW = 5  # Ventana para suavizado de predicciones (majority vote)

# Umbral de confianza para alarmas
FALL_CONFIDENCE_THRESHOLD = 0.7  # Si confianza de FALL > 70%, alarma

# ============================================
# Configuración de visualización
# ============================================
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Colores para cada actividad
ACTIVITY_COLORS = {
    0: '#2ecc71',  # Walking - Verde
    1: '#9b59b6',  # Running - Púrpura    ← NUEVO COLOR
    2: '#3498db',  # Jumping - Azul
    3: '#e74c3c',  # Falling - Rojo
    4: '#f39c12'   # Lying - Naranja
}

# Símbolos emoji para cada actividad (para mostrar en tiempo real)
ACTIVITY_SYMBOLS = {
    'WAL': '🚶',
    'RUN': '🏃',   # ← NUEVO SÍMBOLO
    'JUM': '🤾',
    'FALL': '⚠️',
    'LYI': '🛌'
}

# ============================================
# Validación de configuración
# ============================================
assert NUM_CLASSES == len(LABEL_DICT) == len(CLASS_NAMES) == len(FULL_NAMES), \
    "Inconsistencia en el número de clases"

assert STEP_SIZE * SENSOR_NUM == 120, \
    f"El tamaño de entrada debe ser 120 (STEP_SIZE * SENSOR_NUM), pero es {STEP_SIZE * SENSOR_NUM}"

print("[OK] Configuracion cargada correctamente")
print(f"Numero de clases: {NUM_CLASSES} ({', '.join(FULL_NAMES.values())})")
print(f"Tamano de entrada: {STEP_SIZE} timesteps x {SENSOR_NUM} sensores = {STEP_SIZE * SENSOR_NUM} features")
