import librosa

def Audio2Spectrogram(audio_file, dict_cfg):
    audio, Fs = librosa.load(audio_file)
#    n_mels, hop_length, win_length, n_fft, fmax = mel_cfg
    spectrogram = librosa.feature.melspectrogram(y = audio, sr = Fs,**dict_cfg)
    spectrogram = librosa.power_to_db(spectrogram)
    return librosa.util.normalize(spectrogram)


