import os
import numpy as np
import librosa
from fastapi import UploadFile
from pydub import AudioSegment

save_url = "data/preprocessor/data.csv"
result_url = "data/preprocessor/result.csv"
main_url = "data"
number_labels = 6
number_width_feature = 193
audio_url = "audio.wav"
name_pretrain = "ss.pkl"
fake_db = {
    0: "Đoàn Đình Dũng",
    1: "Nguyễn Việt Hoàng",
    2: "Lưu Hoài Linh",
    3: "Mạc Đình Minh",
    4: "Trịnh Xuân Bách",
    5: "Mai Xuân Minh"
}


def extract_feature(file_name, is_url=True):
    print(file_name['url'])
    X, sample_rate = librosa.load(file_name['url'], res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast, tonnetz


def get_list_dir():
    list_dir = []
    for dirname, dirnames, filenames in os.walk(main_url):
        for dir_file in dirnames:
            list_dir.append(os.path.join(dirname, dir_file))
    return list_dir


def test(url):
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(url)
    return np.concatenate((mfccs, chroma, mel, contrast, tonnetz), axis=0)


def save_audio(file: UploadFile):
    audio = AudioSegment.from_wav(file=file.file)
    audio.export(audio_url, format="wav")
