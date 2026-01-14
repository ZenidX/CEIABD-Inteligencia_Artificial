# Explicación de wav2vec.py

## Descripción General
El archivo `wav2vec.py` contiene funciones para extraer y analizar formantes de vocales a partir de archivos de audio WAV.

---

## Funciones

### 1. `cutvowel(file_address, start, end)`
**Ubicación:** Líneas 6-10

**Propósito:** Extraer un segmento de audio de un archivo WAV

**Parámetros:**
- `file_address` (str): Ruta al archivo WAV
- `start` (float): Tiempo de inicio en segundos
- `end` (float): Tiempo de fin en segundos

**Funcionamiento:**
```python
Fs, audio = sio.wavfile.read(file_address)  # Lee el archivo WAV
# Fs = frecuencia de muestreo (ej: 48000 Hz)
# audio = array con las muestras de audio

audiocut = audio[int(float(start)*Fs):int(float(end)*Fs)]  # Recorta desde start*Fs hasta end*Fs muestras
audiocut = audiocut[:,0]  # Toma solo el canal 0 (convierte estéreo a mono)
return Fs, audiocut
```

**Retorna:**
- `Fs` (int): Frecuencia de muestreo en Hz
- `audiocut` (numpy.ndarray): Segmento de audio recortado (mono)

---

### 2. `wav2vec(cut, Fs)`
**Ubicación:** Líneas 13-49

**Propósito:** Extraer los 3 formantes principales (F1, F2, F3) de un segmento de audio

**Parámetros:**
- `cut` (numpy.ndarray): Segmento de audio (array de muestras)
- `Fs` (int): Frecuencia de muestreo en Hz

**Funcionamiento paso a paso:**

#### Paso 1: FFT (Fast Fourier Transform)
```python
fourierofcut = fft(cut)  # Transforma audio de tiempo → frecuencia
Fsmall = fourierofcut[0:300]  # Toma solo los primeros 300 bins (frecuencias bajas)
```
La FFT convierte la señal temporal en su representación en el dominio de la frecuencia.

#### Paso 2: Calcular magnitud (eliminar fase)
```python
Fsmall = np.sqrt((np.real(Fsmall) ** 2) + np.imag(Fsmall) ** 2)  # Magnitud = √(real² + imag²)
Fsmall[0:30] = 0  # Elimina componentes de muy baja frecuencia (DC y ruido)
```
Solo nos interesa la magnitud (amplitud) de cada frecuencia, no su fase.

#### Paso 3: Filtro de suavizado (Low-Pass Filter)
```python
lpf = 15  # Tamaño del filtro de suavizado
outoffilter = np.zeros(len(Fsmall) - lpf)

for i in range(len(Fsmall) - lpf):
    for j in range(lpf):
        outoffilter[i] += Fsmall[i+j]  # Suma ventana móvil de 15 muestras
```
Aplica un **filtro de promedio móvil** que suaviza el espectro y reduce ruido, facilitando la detección de picos.

#### Paso 4: Detección de 3 picos (formantes)
```python
maxfilt = 25  # Tamaño de la zona a "apagar" alrededor de cada pico

# Primer formante (F1)
Ffon[0] = np.argmax(outoffilter)  # Encuentra el pico máximo
outoffilter[(Ffon[0] - 25):(Ffon[0] + 25)] = 0  # Elimina esa zona

# Segundo formante (F2)
Ffon[1] = np.argmax(outoffilter)  # Encuentra el siguiente máximo
outoffilter[(Ffon[1] - 25):(Ffon[1] + 25)] = 0

# Tercer formante (F3)
Ffon[2] = np.argmax(outoffilter)
outoffilter[(Ffon[2] - 25):(Ffon[2] + 25)] = 0
```
Detecta los 3 picos de mayor amplitud, eliminando una ventana alrededor de cada uno para evitar detecciones duplicadas.

#### Paso 5: Conversión de bins FFT a frecuencias en Hz
```python
Ffon = Ffon * (Fs / len(fourierofcut))  # Fórmula: frecuencia = (bin * Fs) / tamaño_FFT
```
Convierte los índices de los bins de la FFT a frecuencias reales en Hertz.

**Retorna:**
- `Ffon` (numpy.ndarray): Array de 3 valores `[F1, F2, F3]` en Hz (no ordenados necesariamente)

---

### 3. `distancebv(vect1, vect2)`
**Ubicación:** Líneas 52-53

**Propósito:** Calcular la distancia euclidiana entre dos vectores de formantes

**Parámetros:**
- `vect1` (numpy.ndarray): Primer vector de formantes `[F1, F2, F3]`
- `vect2` (numpy.ndarray): Segundo vector de formantes `[F1, F2, F3]`

**Funcionamiento:**
```python
return np.sqrt((vect1[0]-vect2[0])² + (vect1[1]-vect2[1])² + (vect1[2]-vect2[2])²)
```

Calcula la **distancia euclidiana 3D** entre dos puntos en el espacio de formantes:

$$d = \sqrt{(F1_1 - F1_2)^2 + (F2_1 - F2_2)^2 + (F3_1 - F3_2)^2}$$

**Utilidad:** Comparar qué tan similares son dos vocales. Distancias pequeñas indican vocales similares.

**Retorna:**
- `float`: Distancia euclidiana en Hz

---

### 4. `displaycase(testcase, dictmatrix, testmatrix)`
**Ubicación:** Líneas 55-56

**Propósito:** Función incompleta/placeholder

```python
return 1  # No hace nada, solo retorna 1
```

Esta función está **sin implementar** en la versión básica de `wav2vec.py`.

---

## Flujo de Procesamiento

```
┌─────────────┐
│ Audio WAV   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ cutvowel()          │  ← Extrae segmento temporal
│ - Lee archivo WAV   │
│ - Recorta segmento  │
│ - Convierte a mono  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ wav2vec()           │  ← Extrae formantes
│ 1. FFT              │
│ 2. Magnitud         │
│ 3. Suavizado        │
│ 4. Detectar 3 picos │
│ 5. Convertir a Hz   │
└──────┬──────────────┘
       │
       ▼
   [F1, F2, F3]
       │
       ▼
┌─────────────────────┐
│ distancebv()        │  ← Compara vocales
│ - Distancia 3D      │
└─────────────────────┘
```

---

## Parámetros Importantes

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `lpf` | 15 | Tamaño del filtro de suavizado (Low-Pass Filter) |
| `zerofilt` | 15 | Variable declarada pero no utilizada |
| `maxfilt` | 25 | Tamaño de la ventana de supresión alrededor de cada pico |
| Bins FFT | 0:300 | Rango de frecuencias analizado (0-300 bins) |
| Supresión DC | 0:30 | Bins eliminados para quitar componente DC y ruido |

---

## Limitaciones de la versión básica

1. **Parámetros hardcodeados**: Los valores de filtros (15, 25, 300) no son configurables
2. **Sin validación**: No verifica errores en archivos de audio
3. **Ordenamiento**: Los formantes retornados no están ordenados (F1 < F2 < F3)
4. **Rango limitado**: Solo analiza los primeros 300 bins de frecuencia
5. **Función incompleta**: `displaycase()` no está implementada

---

## Versión mejorada

Existe una versión mejorada en `wav2vec_improved.py` que incluye:
- Validación de errores
- Pre-énfasis y ventana Hamming
- Normalización de formantes (log, bark, mel)
- Funciones de visualización completas
- Manejo de NaN
- Documentación detallada

---

## Ejemplo de Uso

```python
from wav2vec import cutvowel, wav2vec, distancebv

# Extraer segmento de audio
Fs, audio_segment = cutvowel("vowels/alex.wav", start=2.285, end=2.329)

# Obtener formantes
formants = wav2vec(audio_segment, Fs)
print(f"Formantes: F1={formants[0]:.0f} Hz, F2={formants[1]:.0f} Hz, F3={formants[2]:.0f} Hz")

# Comparar dos vocales
distance = distancebv(formants1, formants2)
print(f"Distancia entre vocales: {distance:.2f} Hz")
```

---

## Fundamentos Teóricos

### ¿Qué son los formantes?

Los **formantes** son las frecuencias de resonancia del tracto vocal humano. Cada vocal tiene un patrón característico de formantes:

- **F1**: Relacionado con la altura de la lengua (vocales abiertas vs cerradas)
- **F2**: Relacionado con la posición anterior/posterior de la lengua
- **F3**: Menos variable, relacionado con el redondeo de los labios

### Rangos típicos de formantes (adultos):

| Vocal | F1 (Hz) | F2 (Hz) | F3 (Hz) |
|-------|---------|---------|---------|
| /a/   | 700-900 | 1200-1400 | 2500 |
| /e/   | 400-600 | 1800-2200 | 2500 |
| /i/   | 250-350 | 2200-2800 | 3000 |
| /o/   | 400-600 | 800-1000 | 2500 |
| /u/   | 250-350 | 600-800 | 2200 |
