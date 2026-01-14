import numpy as np
import scipy.io as sio
from numpy.fft import fft
import matplotlib.pyplot as plt


def cutvowel(file_address, start, end):
    """
    Extrae un segmento de audio de un archivo WAV.

    Args:
        file_address: Ruta al archivo WAV
        start: Tiempo de inicio en segundos
        end: Tiempo de fin en segundos

    Returns:
        Fs: Frecuencia de muestreo
        audiocut: Segmento de audio extraído
    """
    try:
        Fs, audio = sio.wavfile.read(file_address)
    except Exception as e:
        raise ValueError(f"Error leyendo archivo '{file_address}': {e}")

    start_sample = int(float(start) * Fs)
    end_sample = int(float(end) * Fs)

    # Validar rango
    if start_sample < 0 or end_sample > len(audio):
        raise ValueError(f"Rango [{start}, {end}] fuera de límites del audio (duración: {len(audio)/Fs:.2f}s)")

    if start_sample >= end_sample:
        raise ValueError("El tiempo de inicio debe ser menor que el final")

    audiocut = audio[start_sample:end_sample]

    # Manejar tanto mono como estéreo
    if len(audiocut.shape) > 1:
        audiocut = audiocut[:, 0]  # Tomar primer canal si es estéreo

    if len(audiocut) == 0:
        raise ValueError("El segmento de audio está vacío")

    return Fs, audiocut


def wav2vec(cut, Fs):
    """
    Extrae los tres primeros formantes (F1, F2, F3) de un segmento de audio.

    Args:
        cut: Segmento de audio (array numpy)
        Fs: Frecuencia de muestreo en Hz

    Returns:
        Ffon: Array con las frecuencias de los tres formantes en Hz
    """
    if len(cut) == 0:
        raise ValueError("El segmento de audio está vacío")

    # Pre-énfasis para resaltar frecuencias altas (típico en análisis de voz)
    pre_emphasis = 0.97
    emphasized = np.append(cut[0], cut[1:] - pre_emphasis * cut[:-1])

    # Aplicar ventana de Hamming para reducir efectos de borde
    window = np.hamming(len(emphasized))
    emphasized = emphasized * window

    # Transformada de Fourier
    fourierofcut = fft(emphasized)

    # Rango de análisis adaptativo (formantes vocálicos típicamente < 4000 Hz)
    max_freq = 4000  # Hz
    max_bin = int(max_freq * len(fourierofcut) / Fs)
    max_bin = min(max_bin, len(fourierofcut) // 2)  # No exceder Nyquist

    Fsmall = fourierofcut[0:max_bin]

    # Calcular magnitud del espectro
    Fsmall = np.sqrt((np.real(Fsmall) ** 2) + np.imag(Fsmall) ** 2)

    # Filtrar frecuencias muy bajas (< ~200 Hz para excluir pitch fundamental)
    min_freq = 200
    min_bin = int(min_freq * len(fourierofcut) / Fs)
    Fsmall[0:min_bin] = 0

    # Suavizado con ventana móvil adaptativa (~50Hz de ancho)
    window_hz = 50
    window_size = max(3, int(window_hz * len(Fsmall) / max_freq))
    if window_size % 2 == 0:  # Asegurar ventana impar
        window_size += 1

    smoothed = np.convolve(Fsmall, np.ones(window_size) / window_size, mode='same')

    # Encontrar los 3 picos más prominentes
    filter_hz = 150  # Separación mínima entre formantes en Hz
    filter_width = int(filter_hz * len(smoothed) / max_freq)
    filter_width = max(filter_width, 5)  # Mínimo 5 bins

    Ffon = np.zeros(3, dtype=np.int64)
    temp_spectrum = smoothed.copy()

    for i in range(3):
        if np.max(temp_spectrum) == 0:
            # No hay más picos significativos
            Ffon[i] = 0
            continue

        Ffon[i] = np.argmax(temp_spectrum)

        # Enmascarar región alrededor del pico encontrado
        start_mask = max(0, Ffon[i] - filter_width)
        end_mask = min(len(temp_spectrum), Ffon[i] + filter_width)
        temp_spectrum[start_mask:end_mask] = 0

    # Convertir índices de bins a frecuencias en Hz
    Ffon = Ffon * (Fs / len(fourierofcut))

    # Ordenar formantes de menor a mayor frecuencia
    Ffon.sort()

    # Filtrar formantes que quedaron en 0
    Ffon = Ffon[Ffon > 0]

    # Rellenar con NaN si no se encontraron 3 formantes
    if len(Ffon) < 3:
        Ffon = np.append(Ffon, [np.nan] * (3 - len(Ffon)))

    return Ffon


def distancebv(vect1, vect2):
    """
    Calcula la distancia euclidiana entre dos vectores de formantes.
    Ignora valores NaN en el cálculo.

    Args:
        vect1: Primer vector de formantes
        vect2: Segundo vector de formantes

    Returns:
        Distancia euclidiana
    """
    # Crear máscaras para valores válidos (no NaN)
    mask = ~(np.isnan(vect1) | np.isnan(vect2))

    if not np.any(mask):
        return np.inf  # Si no hay valores válidos, distancia infinita

    # Calcular distancia solo con valores válidos
    diff = vect1[mask] - vect2[mask]
    return np.sqrt(np.sum(diff ** 2))


def normalize_formants(formants, method='log'):
    """
    Normaliza formantes para reducir variabilidad entre hablantes.

    Args:
        formants: Array de frecuencias formantes en Hz
        method: Método de normalización ('log', 'bark', 'none')

    Returns:
        Formantes normalizados
    """
    formants = np.array(formants, dtype=float)

    if method == 'log':
        # Transformación logarítmica (común en fonética)
        return np.log(formants + 1e-10)  # +epsilon para evitar log(0)
    elif method == 'bark':
        # Escala de Bark (percepción auditiva)
        return 26.81 * formants / (1960 + formants) - 0.53
    elif method == 'mel':
        # Escala Mel
        return 2595 * np.log10(1 + formants / 700)
    else:
        return formants


def displaycase(testcase, dictmatrix, testmatrix, labels=None, title="Análisis de Formantes"):
    """
    Visualiza la comparación entre un caso de prueba y el diccionario de formantes.

    Args:
        testcase: Vector de formantes del caso de prueba [F1, F2, F3]
        dictmatrix: Matriz de formantes del diccionario (N x 3 o más)
        testmatrix: Matriz de casos de prueba (M x 3 o más)
        labels: Etiquetas para los puntos del diccionario (opcional)
        title: Título de la figura

    Returns:
        Figura de matplotlib
    """
    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Espacio F1-F2
    plt.subplot(1, 3, 1)
    if len(dictmatrix) > 0:
        if labels is not None and len(labels) == len(dictmatrix):
            # Si hay etiquetas, colorear por categoría
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(dictmatrix[mask, 0], dictmatrix[mask, 1],
                          c=[colors[i]], label=str(label), alpha=0.6, s=50)
        else:
            plt.scatter(dictmatrix[:, 0], dictmatrix[:, 1],
                       c='blue', label='Diccionario', alpha=0.6, s=50)

    if len(testmatrix) > 0:
        plt.scatter(testmatrix[:, 0], testmatrix[:, 1],
                   c='red', marker='x', s=100, label='Test', linewidths=2)

    plt.scatter(testcase[0], testcase[1],
               c='green', marker='*', s=300, label='Caso actual',
               edgecolors='black', linewidths=1.5)

    plt.xlabel('F1 (Hz)', fontsize=11)
    plt.ylabel('F2 (Hz)', fontsize=11)
    plt.title('Espacio F1-F2', fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Espacio F2-F3
    plt.subplot(1, 3, 2)
    if len(dictmatrix) > 0 and dictmatrix.shape[1] >= 3:
        if labels is not None and len(labels) == len(dictmatrix):
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(dictmatrix[mask, 1], dictmatrix[mask, 2],
                          c=[colors[i]], label=str(label), alpha=0.6, s=50)
        else:
            plt.scatter(dictmatrix[:, 1], dictmatrix[:, 2],
                       c='blue', label='Diccionario', alpha=0.6, s=50)

    if len(testmatrix) > 0 and testmatrix.shape[1] >= 3:
        plt.scatter(testmatrix[:, 1], testmatrix[:, 2],
                   c='red', marker='x', s=100, label='Test', linewidths=2)

    if len(testcase) >= 3:
        plt.scatter(testcase[1], testcase[2],
                   c='green', marker='*', s=300, label='Caso actual',
                   edgecolors='black', linewidths=1.5)

    plt.xlabel('F2 (Hz)', fontsize=11)
    plt.ylabel('F3 (Hz)', fontsize=11)
    plt.title('Espacio F2-F3', fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Subplot 3: Gráfico de barras comparativo
    plt.subplot(1, 3, 3)
    x = np.arange(3)
    width = 0.35

    # Barras del caso de prueba
    valid_testcase = testcase[~np.isnan(testcase)]
    plt.bar(x[:len(valid_testcase)] - width/2, valid_testcase, width,
           label='Caso de prueba', color='green', alpha=0.7)

    # Barras del promedio del diccionario
    if len(dictmatrix) > 0:
        mean_dict = np.nanmean(dictmatrix[:, :3], axis=0)
        valid_mean = mean_dict[~np.isnan(mean_dict)]
        plt.bar(x[:len(valid_mean)] + width/2, valid_mean, width,
               label='Promedio diccionario', color='blue', alpha=0.7)

    plt.xlabel('Formante', fontsize=11)
    plt.ylabel('Frecuencia (Hz)', fontsize=11)
    plt.title('Valores de formantes', fontsize=12, fontweight='bold')
    plt.xticks(x, ['F1', 'F2', 'F3'])
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_spectrum(cut, Fs, formants=None, title="Espectro de frecuencias"):
    """
    Visualiza el espectro de frecuencias del audio con los formantes marcados.

    Args:
        cut: Segmento de audio
        Fs: Frecuencia de muestreo
        formants: Frecuencias formantes a marcar (opcional)
        title: Título del gráfico
    """
    # Pre-énfasis
    pre_emphasis = 0.97
    emphasized = np.append(cut[0], cut[1:] - pre_emphasis * cut[:-1])

    # Ventana de Hamming
    window = np.hamming(len(emphasized))
    emphasized = emphasized * window

    # FFT
    fourierofcut = fft(emphasized)
    N = len(fourierofcut)

    # Calcular magnitud y convertir a dB
    magnitude = np.abs(fourierofcut[:N//2])
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Eje de frecuencias
    freqs = np.fft.fftfreq(N, 1/Fs)[:N//2]

    # Graficar
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, magnitude_db, linewidth=0.8, color='blue', alpha=0.7)

    # Marcar formantes si se proporcionan
    if formants is not None:
        for i, f in enumerate(formants):
            if not np.isnan(f) and f > 0:
                plt.axvline(x=f, color='red', linestyle='--', linewidth=2,
                           label=f'F{i+1} = {f:.0f} Hz', alpha=0.8)

    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Magnitud (dB)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, 4000)
    plt.grid(True, alpha=0.3)
    if formants is not None:
        plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    return plt.gcf()


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("WAV2VEC - Extractor de Formantes Mejorado")
    print("=" * 60)
    print("\nEste script extrae formantes (F1, F2, F3) de vocales en audio.")
    print("\nFunciones disponibles:")
    print("  - cutvowel(): Extrae segmentos de audio")
    print("  - wav2vec(): Calcula formantes de un segmento")
    print("  - distancebv(): Calcula distancia entre vectores de formantes")
    print("  - normalize_formants(): Normaliza formantes")
    print("  - displaycase(): Visualiza comparación de formantes")
    print("  - plot_spectrum(): Muestra espectro con formantes marcados")
    print("\n" + "=" * 60)
