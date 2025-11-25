# Real-Time Human Activity Recognition (HAR) System

Sistema de reconocimiento de actividades humanas en tiempo real usando MPU 6050 con Arduino Uno o ESP32.

## 🎯 Características

- **Detección en tiempo real** de 4 actividades:
  - 🚶 **WAL** - Caminando (Walking)
  - 🤾 **JUM** - Saltando (Jumping)
  - ⚠️ **FALL** - Cayendo (Falling)
  - 🛌 **LYI** - Acostado (Lying)

- **Sliding Window Approach**: Ventana deslizante de 20 timesteps para procesamiento en tiempo real
- **MLP Neural Network**: Red neuronal multicapa con 2 capas ocultas
- **Compatible con Arduino Uno y ESP32**

## 📋 Requisitos

### Hardware
- Arduino Uno o ESP32
- MPU 6050 (GY-521)
- Cables Dupont
- Cable USB

### Software
- Python 3.8+
- Arduino IDE
- Bibliotecas listadas en `requirements.txt`

## 🔧 Conexión MPU 6050

```
MPU 6050  ->  Arduino/ESP32
VCC       ->  5V (3.3V para ESP32)
GND       ->  GND
SCL       ->  A5 (SCL)
SDA       ->  A4 (SDA)
```

## 📦 Instalación

### 1. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

### 2. Cargar código Arduino

1. Abrir `arduino_mpu6050/arduino_mpu6050.ino` en Arduino IDE
2. Seleccionar board: Arduino Uno o ESP32
3. Seleccionar puerto COM correcto
4. Subir el sketch

### 3. Verificar puerto serial

En Windows, revisar en **Device Manager** → **Ports (COM & LPT)**

Editar `config.py` y cambiar:
```python
SERIAL_PORT = 'COM3'  # Cambiar por tu puerto
```

## 🚀 Uso

### Paso 1: Generar datos sintéticos

```bash
python generate_synthetic_data.py
```

Esto genera secuencias de:
- Caminando (5s) → Saltando (2s) → Cayendo (0.5s) → Acostado (3s) → Caminando (5s)

Los datos se guardan en `./dataset/`

### Paso 2: Entrenar el modelo

```bash
python train_model.py
```

Entrena un modelo MLP y lo guarda en `./model_har/`

Resultados:
- Matriz de confusión: `./results/confusion_matrix.png`
- Historial de entrenamiento: `./results/training_history.png`

### Paso 3: Reconocimiento en tiempo real

```bash
python realtime_har.py
```

O especificar puerto:
```bash
python realtime_har.py COM5
```

El sistema mostrará la actividad actual en tiempo real:

```
[  245] 🚶 Activity: WAL  | Confidence:  95.3%
[  246] 🚶 Activity: WAL  | Confidence:  96.1%
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
>>> FALL DETECTED! <<<
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[  247] ⚠️ Activity: FALL | Confidence:  97.8%
[  248] 🛌 Activity: LYI  | Confidence:  94.2% (Person is lying down)
```

## 📊 Arquitectura del Sistema

### Sliding Window Approach

El sistema usa una ventana deslizante de 20 timesteps (~400ms a 50Hz):

```
[Sample 0] [Sample 1] ... [Sample 19]  → Prediction 1
           [Sample 1] [Sample 2] ... [Sample 20]  → Prediction 2
                      [Sample 2] [Sample 3] ... [Sample 21]  → Prediction 3
```

Esto convierte el problema de series temporales en clasificación, permitiendo predicciones rápidas.

### Modelo MLP

```
Input: (20 timesteps × 6 features) = 120 features
  ↓
Flatten
  ↓
Dense(128) + ReLU + Dropout(0.3)
  ↓
Dense(128) + ReLU + Dropout(0.3)
  ↓
Dense(4) + Softmax
  ↓
Output: [WAL, JUM, FALL, LYI]
```

## 🔄 Entrenar con tus propios datos

### Opción 1: Recolectar datos desde Arduino

1. Modificar `realtime_har.py` para guardar datos en CSV
2. Realizar las actividades con etiquetas
3. Guardar en `./dataset/` con formato:

```csv
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,label
0.123,0.987,0.045,12.5,-3.2,8.1,WAL
...
```

### Opción 2: Usar datos del profesor

Extraer `Example Datasets.zip` en `./dataset/` y ejecutar:

```bash
python train_model.py
```

## 📁 Estructura del Proyecto

```
tarea_De_movimiento/
├── arduino_mpu6050/
│   └── arduino_mpu6050.ino      # Código Arduino/ESP32
├── config.py                     # Configuración del sistema
├── generate_synthetic_data.py    # Generador de datos sintéticos
├── train_model.py                # Entrenamiento del modelo
├── realtime_har.py               # Predicción en tiempo real
├── requirements.txt              # Dependencias Python
├── dataset/                      # Datos de entrenamiento
├── model_har/                    # Modelo entrenado
└── results/                      # Gráficas y resultados
```

## 🎓 Basado en el ejemplo del profesor

Este sistema está inspirado en:
- **Realtime Fall Detection and HAR Using MLP**
- Usa sliding window de 20 timesteps
- Red neuronal MLP para clasificación
- Comunicación serial para datos en tiempo real

## 🛠️ Troubleshooting

### Error: No serial port found
- Verificar que Arduino esté conectado
- Revisar puerto COM en Device Manager
- Actualizar `SERIAL_PORT` en `config.py`

### El modelo no detecta bien las actividades
- Recolectar más datos reales
- Ajustar `STEP_SIZE` en `config.py`
- Aumentar `EPOCHS` en entrenamiento

### Predicciones inestables
- El sistema ya incluye suavizado (majority vote de últimas 5 predicciones)
- Aumentar `prediction_history.maxlen` en `realtime_har.py`

## 📝 Notas

- Frecuencia de muestreo: 50 Hz
- Formato de datos Arduino: `!ax,ay,az,gx,gy,gz@`
- El modelo usa **sparse categorical crossentropy** (labels como enteros)
- Dropout de 0.3 para prevenir overfitting

## 🔮 Futuras mejoras

- [ ] Agregar más actividades (correr, sentarse, etc.)
- [ ] Implementar LSTM para mejor captura temporal
- [ ] Crear interfaz gráfica en tiempo real
- [ ] Guardar logs de actividades detectadas
- [ ] Alertas por email/SMS en caso de caída

---

**Autor**: Proyecto ML_LPTM  
**Fecha**: 2025  
**Basado en**: Realtime Fall Detection Using MLP
