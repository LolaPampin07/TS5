"""
En el repositorio PDStestbench encontrará tres tipos de señales registradas:

Electrocardiograma (ECG). En el archivo ECG_TP4.mat encontrará un registro electrocardiográfico (ECG) registrado durante una prueba de esfuerzo, junto con una serie de variables descriptas más abajo.
Pletismografía (PPG). El archivo PPG.csv contiene una señal registrada en reposo de un estudiante de la materia que ha donado su registro para esta actividad.
Audio. Tres registros en los que el profesor pronuncia una frase, y otros dos en los que se silba una melodía muy conocida.
Los detalles de cómo acceder a dichos registros los pueden encontrar en lectura_sigs.py

Se pide:

1) Realizar la estimación de la densidad espectral de potencia (PSD) de cada señal mediante alguno de los métodos vistos en clase (Periodograma ventaneado, Welch, Blackman-Tukey).

2) Realice una estimación del ancho de banda de cada señal y presente los resultados en un tabla para facilitar la comparación.

Bonus:
"""
# %% Imports

import numpy as np
from scipy import signal as sig
from scipy.signal import periodogram , get_window
from numpy.fft import fft
import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

# %% Generacion de senales + graficos


#% ECG

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
# sio.whosmat('ECG_TP4.mat')
# mat_struct = sio.loadmat('./ECG_TP4.mat')

# ecg_one_lead = mat_struct['ecg_lead']
# N = len(ecg_one_lead)

# hb_1 = mat_struct['heartbeat_pattern1']
# hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])
# plt.title('ecg con ruido')

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure()
plt.plot(ecg_one_lead)
plt.title('ECG sin ruido')


#% PPG

####################################
# Lectura de pletismografía (PPG)  #
#senal proporcional a la cant de oxigeno que tenes en sangre en el tejido capilar
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

plt.figure()
plt.plot(ppg)
plt.title('PPG sin ruido')


#% AUDIO

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)
plt.title('La cucaracha')

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)

# %% ESTIMACION
# ppg x welch y la cucaracha x bt

# ECG POR PERIODOGRAMA

win_ecg = get_window('hann', len(ecg_one_lead))
ecg_ventaneado=ecg_one_lead*win_ecg

f_ecg, Pxx_ecg = periodogram(ecg_ventaneado, fs_ecg)

plt.figure()
plt.plot(f_ecg,10*np.log10(Pxx_ecg)**2)
plt.title("ECG (Periodograma ventaneado)")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad Espectral de Potencia [dB]')
plt.grid(True)
plt.show()

# rxx_ecg= sig.correlate(ecg_one_lead,ecg_one_lead, mode='full') #autocorrelacion del ECG
# # plt.figure()
# # plt.plot(rxx_ecg)
# # plt.title('autocorrelacion ecg')

# sxx_ecg= (np.abs(fft(rxx_ecg))**2)/len(ecg_one_lead)

# plt.figure(figsize=(10,20))
# plt.plot(np.log10(sxx_ecg)*10)
# plt.title('Densidad Espectral de Potencia ECG (Periodograma)')

# PPG por welch

# PARAMETROS WELCH

cant_promedios_ppg = 20 #cambia mucho la forma, cuanto mas chico mas varianza
nperseg_ppg = len(ppg) // cant_promedios_ppg
nfft_ppg = 2 * nperseg_ppg
win_ppg = "hamming"


f_ppg, Pxx_ppg = sig.welch(ppg, fs=fs_ppg, window = win_ppg, nperseg=nperseg_ppg, nfft=nfft_ppg)

"""
sig.welch:
    - Divide la señal en segmentos (con posible solapamiento, si se especifica).
    - Aplica la ventana a cada segmento.
    - Calcula la FFT de cada segmento.
    - Promedia los espectros de potencia de todos los segmentos.
"""

#Gráfico de la PSD - PPG
plt.figure(figsize=(10,20))
plt.plot(f_ppg, 10*np.log10(Pxx_ppg)**2)
plt.title("PPG (Welch)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Densidad espectral de potencia [dB]')
plt.grid(True)
plt.tight_layout()
plt.xlim([0, 30]) #como es pasabajos, limito
plt.show()

#%% Audio por Blackman Tukey
# # Configuración e inicio de la simulación

def blackman_tukey(x,  M = None):    
    
    N = len(x)

    if M is None:
        M = N//20 # Ventana mas chica
    M = min(M, N // 2 - 1)

    x = np.asarray(x)
    r_full = np.correlate(x, x, mode='full') / N
    mid = len(r_full) // 2
    r = r_full[mid - (M - 1) : mid + M]

    window = sig.windows.blackman(len(r))
    r_windowed = r * window

    Px = np.abs(np.fft.fft(r_windowed, n=N))
    return Px

# PSD del audio
Pxx_audio = blackman_tukey(wav_data, len(wav_data)//2)
f_audio = np.fft.fftfreq(len(wav_data), d=1/fs_audio)

# Me quedo con solo la mitad positiva del espectro
half = len(wav_data) // 2
freqs_pos = f_audio[:half]
Pxx_audio_pos = (2 / (fs_audio * len(wav_data))) * Pxx_audio[:half]
Pxx_audio_db = 10 * np.log10(Pxx_audio_pos)

plt.figure()
plt.plot(freqs_pos, Pxx_audio_db)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.title("Audio (Blackman-Tukey)")
# plt.xlim(-100, 4000)
plt.grid(True)
plt.show()

# %% Calculo de frecuencia de corte y ancho de banda
# Idea general:
# 1) Se calcula la energía acumulada normalizada de la PSD.
# 2) Se busca el índice donde la energía acumulada alcanza el 99% del total.
# 3) La frecuencia correspondiente a ese punto se define como la frecuencia de corte (fc).

# Todas las señales utilizadas (ECG, PPG y audio) son pasabajo o limitadas en frecuencia,
# es decir que concentran la mayor parte de su energía en las bajas frecuencias y su contenido en altas frecuencias disminuye.
# Por lo tanto, el ancho de banda efectivo (BW) puede aproximarse mediante la frecuencia de corte obtenida: BW ≈ fc

# -------- ECG --------
df_ecg = f_ecg[1] - f_ecg[0]
energia_acum_ecg = np.cumsum(Pxx_ecg) * df_ecg
energia_acum_ecg_norm = energia_acum_ecg / energia_acum_ecg[-1]
indice_corte_ecg = np.where(energia_acum_ecg_norm >= 0.99)[0][0]
frecuencia_corte_ecg = f_ecg[indice_corte_ecg]

# Grafico
plt.figure()
plt.plot(f_ecg, 10*np.log10(Pxx_ecg)**2, label = 'PSD del ECG')
plt.axvline(frecuencia_corte_ecg, color='orange', linestyle='--', label=f'Frecuencia de corte = {frecuencia_corte_ecg:.2f} Hz')
plt.title("PSD ECG + Frecuencia de corte (99%)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia")
plt.legend()
plt.grid(True)
plt.show()

# -------- PPG --------
df_ppg = f_ppg[1] - f_ppg[0]
energia_acum_ppg = np.cumsum(Pxx_ppg) * df_ppg
energia_acum_ppg_norm = energia_acum_ppg / energia_acum_ppg[-1]
indice_corte_ppg = np.where(energia_acum_ppg_norm >= 0.99)[0][0]
frecuencia_corte_ppg = f_ppg[indice_corte_ppg]

# Grafico
plt.figure()
plt.plot(f_ppg, 10*np.log10(Pxx_ppg)**2, label = 'PSD del PPG')
plt.axvline(frecuencia_corte_ppg, color='orange', linestyle='--', label=f'Frecuencia de corte = {frecuencia_corte_ppg:.2f} Hz')
plt.title("PSD PPG + Frecuencia de corte (99%)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia")
plt.xlim(0, 40)
plt.legend()
plt.grid(True)
plt.show()

# -------- AUDIO --------
# Calculo frecuencia de corte (99%)
df_audio = freqs_pos[1] - freqs_pos[0]
energia_acum_audio = np.cumsum(Pxx_audio_pos) * df_audio
energia_acum_audio_norm = energia_acum_audio / energia_acum_audio[-1]
indice_corte_audio = np.where(energia_acum_audio_norm >= 0.99)[0][0]
frecuencia_corte_audio = freqs_pos[indice_corte_audio]

plt.figure()
plt.plot(freqs_pos, 10*np.log10(Pxx_audio_pos + 1e-12), label='PSD del audio')
plt.axvline(frecuencia_corte_audio, color='orange', linestyle='--', label=f'Frecuencia de corte = {frecuencia_corte_audio:.2f} Hz')
plt.title("PSD AUDIO + Frecuencia de corte (99%)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [dB]")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)
plt.show()

# %% Tabla con los resultados del ancho de banda
data = [
    ["ECG", f"{frecuencia_corte_ecg:.2f} Hz"],
    ["PPG", f"{frecuencia_corte_ppg:.2f} Hz"],
    ["AUDIO (La cucaracha)", f"{frecuencia_corte_audio:.2f} Hz"],
]

fig, ax = plt.subplots(figsize = (20, 1.5))
ax.axis('off')

tabla = ax.table(cellText = data, colLabels = ["Señal", "Ancho de banda"], cellLoc = 'center', loc = 'center')

# Esto es para poner en negrita los "titulos"
tabla[0,0].get_text().set_fontweight('bold')
tabla[0,1].get_text().set_fontweight('bold')

tabla.scale(1, 1.5)
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)

plt.title("RESULTADOS", fontweight = 'bold', pad=10)
plt.show()

# %%
from scipy.io import wavfile
from scipy.signal import get_window, periodogram
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo WAV
fs_audio, wav_data = wavfile.read(r'C:\Users\Usuario\Documents\Cata\Unsam\3er año\APS\TS5\la cucaracha.wav')

# Aplicar ventana
win_audio = get_window('hann', len(wav_data))
audio_ventaneado = wav_data * win_audio

# Calcular periodograma
f_audio, Pxx_audio = periodogram(audio_ventaneado, fs_audio)

# Graficar
plt.figure()
plt.plot(f_audio, 10 * np.log10(Pxx_audio))
plt.title("Audio (Periodograma ventaneado)")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad Espectral de Potencia [dB]')
plt.grid(True)
plt.show()

# Frecuencia de corte
df_audio = f_audio[1] - f_audio[0]
energia_acum_audio = np.cumsum(Pxx_audio) * df_audio
energia_acum_audio_norm = energia_acum_audio / energia_acum_audio[-1]
indice_corte_audio = np.where(energia_acum_audio_norm >= 0.99)[0][0]
frecuencia_corte_audio = f_audio[indice_corte_audio]

# Grafico
plt.figure()
plt.plot(f_audio, Pxx_audio, label = 'PSD de la cucaracha')
plt.axvline(frecuencia_corte_audio, color='orange', linestyle='--', label=f'Frecuencia de corte = {frecuencia_corte_audio:.2f} Hz')
plt.title("PSD de la cucaracha + Frecuencia de corte (99%)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)
plt.show()

# %%

# PARAMETROS WELCH

cant_promedios_audio = 20 #cambia mucho la forma, cuanto mas chico mas varianza
ruta_audio = r'C:\Users\Usuario\Documents\Cata\Unsam\3er año\APS\TS5\la cucaracha.wav'
fs_audio, audio = wavfile.read(ruta_audio)
nperseg_audio = len(audio) // cant_promedios_audio
nfft_audio = 2 * nperseg_audio
win_audio = "hamming"


f_audio, Pxx_audio = sig.welch(audio, fs=fs_audio, window = win_audio, nperseg=nperseg_audio, nfft=nfft_audio)

"""
sig.welch:
    - Divide la señal en segmentos (con posible solapamiento, si se especifica).
    - Aplica la ventana a cada segmento.
    - Calcula la FFT de cada segmento.
    - Promedia los espectros de potencia de todos los segmentos.
"""

#Gráfico de la PSD - PPG
plt.figure(figsize=(10,20))
plt.plot(f_audio, 10 * np.log10(Pxx_audio))
plt.title("La cucaracha (Welch)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Densidad espectral de potencia [dB]')
plt.grid(True)
plt.tight_layout()
plt.show()


# Frecuencia de corte
df_audio = f_audio[1] - f_audio[0]
energia_acum_audio = np.cumsum(Pxx_audio) * df_audio
energia_acum_audio_norm = energia_acum_audio / energia_acum_audio[-1]
indice_corte_audio = np.where(energia_acum_audio_norm >= 0.99)[0][0]
frecuencia_corte_audio = f_audio[indice_corte_audio]

# Grafico
plt.figure()
plt.plot(f_audio, Pxx_audio, label = 'PSD de la cucaracha')
plt.axvline(frecuencia_corte_audio, color='orange', linestyle='--', label=f'Frecuencia de corte = {frecuencia_corte_audio:.2f} Hz')
plt.title("PSD de la cucaracha + Frecuencia de corte (99%)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)
plt.show()


