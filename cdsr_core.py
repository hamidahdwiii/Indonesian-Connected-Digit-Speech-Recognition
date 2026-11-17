"""
cdsr_core.py — Modul inti Indonesian Connected Digit Speech Recognition (CDSR).

Satu sumber kode untuk NOTEBOOK (pelatihan & evaluasi) dan DASHBOARD (inferensi),
sehingga cara audio diproses saat latih dan saat dipakai dijamin sama persis.

Versi 2 (tahan derau). Perubahan utama dibanding versi 1:
  * preprocess(): high-pass -> PENJERNIHAN (spectral gating / Wiener) -> deteksi suara
    berbasis noise floor -> normalisasi kekerasan. Dipakai SAAT LATIH dan SAAT PREDIKSI.
  * augment(): menambah derau, gema ruangan, efek mikrofon, perubahan kecepatan dan
    jeda hening di awal/akhir pada audio latih, agar model terbiasa dengan rekaman
    mikrofon laptop/HP (data asli hanya rekaman studio).
  * CTCNet: CNN 2D + BiLSTM + CTC (lebih kuat dari CNN 1D versi 1).
  * Recognizer.recognize() juga mengembalikan perkiraan SNR, peringatan kualitas,
    dan audio hasil penjernihan (bisa diputar di dashboard).

Isi modul:
  1. Kosakata & parsing transkrip dari nama file   -> parse_filename()
  2. Pemuatan & penjernihan audio                   -> load_audio(), preprocess()
  3. Augmentasi derau (khusus pelatihan)            -> augment(), NoiseBank
  4. Fitur model CTC (log-mel per frame)            -> ctc_features()
  5. Segmentasi kata + fitur per segmen             -> segment_utterance(), segment_features()
  6. Model CNN-BiLSTM-CTC (PyTorch) + decoding      -> CTCNet, ctc_greedy_decode()
  7. Metrik urutan (WER/SER, alignment)             -> align(), error_counts()
  8. Recognizer: memuat artefak & mengenali audio   -> Recognizer
"""
from __future__ import annotations

import io
import json
import os
import re
import time

import joblib
import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

# =============================================================================
# 1. KOSAKATA & PARSING TRANSKRIP
# =============================================================================
SR = 16000                     # sample rate dataset (16 kHz)
# 11 token: '0' diucapkan "nol", 'K' diucapkan "kosong" (keduanya bernilai 0)
TOKENS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'K']
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}
TOKEN_WORDS = {'0': 'nol', '1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat',
               '5': 'lima', '6': 'enam', '7': 'tujuh', '8': 'delapan',
               '9': 'sembilan', 'K': 'kosong'}

# Nama file: <PENUTUR>_<URUTAN DIGIT><PENANDA REKAMAN>, mis. MWS_985K3A
_NAME_RE = re.compile(r'^([A-Za-z]{3})_([0-9K]+)([A-Z]?)$')


def parse_filename(file_name: str):
    """Mengurai nama file menjadi (penutur, gender, daftar token).

    Contoh: 'MWS_985K3A.wav' -> ('MWS', 'M', ['9','8','5','K','3']).
    Mengembalikan None bila nama tidak sesuai pola.
    """
    stem = os.path.splitext(os.path.basename(file_name))[0]
    m = _NAME_RE.match(stem)
    if not m:
        return None
    speaker = m.group(1).upper()
    return speaker, speaker[0], list(m.group(2))


def to_value(tokens):
    """Nilai angka yang dipakai layanan: 'K' (kosong) dan '0' (nol) sama-sama 0."""
    return ['0' if t == 'K' else t for t in tokens]


def to_words(tokens):
    """Kalimat konfirmasi, mis. ['9','1','K'] -> 'sembilan satu kosong'."""
    return ' '.join(TOKEN_WORDS[t] for t in tokens)


# =============================================================================
# 2. PEMUATAN & PENJERNIHAN AUDIO
# =============================================================================
def load_audio(source) -> np.ndarray:
    """Memuat audio (path, bytes, atau file-like) -> mono 16 kHz, puncak dinormalkan ke 1.

    Rekaman browser biasanya 44,1/48 kHz dan kadang stereo; keduanya diseragamkan di sini.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, _ = librosa.load(source, sr=SR, mono=True)
    y = np.nan_to_num(y).astype(np.float32)
    peak = np.max(np.abs(y)) if y.size else 0.0
    return (y / peak).astype(np.float32) if peak > 0 else y


_HP_SOS = signal.butter(4, 70, 'highpass', fs=SR, output='sos')   # buang dengung AC/meja
DN_FFT, DN_HOP = 512, 128
DN_FLOOR_DB = -18.0          # redaman maksimum penjernihan (lebih dalam -> artefak "musical noise")
DN_OVERSUB = 1.5             # faktor pengurangan derau (>1 = lebih agresif)


def frame_db(y: np.ndarray, frame=400, hop=160) -> np.ndarray:
    """Energi per frame (dB)."""
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    return 20 * np.log10(rms + 1e-8)


def estimate_snr(y: np.ndarray) -> float:
    """Perkiraan SNR (dB): energi bagian bersuara vs 15% frame paling senyap.

    Akurat (sekitar 1-2 dB) untuk derau STABIL: kipas, AC, dengung, desis mikrofon.
    Untuk derau yang naik-turun, terutama suara orang lain, SNR cenderung terbaca jauh lebih
    tinggi (lebih "jernih") daripada kenyataannya karena jeda dalam derau itu ikut terhitung.
    """
    if len(y) < 800:
        return 0.0
    p = 10 ** (frame_db(y) / 10)
    p_sorted = np.sort(p)
    noise = p_sorted[:max(3, int(0.15 * len(p)))].mean()
    speech = p_sorted[-max(3, int(0.30 * len(p))):].mean()
    return float(10 * np.log10(max(speech / (noise + 1e-12) - 1, 1e-3)))


def denoise(y: np.ndarray, oversub: float = DN_OVERSUB, floor_db: float = DN_FLOOR_DB) -> np.ndarray:
    """Penjernihan suara dengan spectral gating (penapis Wiener sederhana).

    1. Profil derau per frekuensi = rata-rata 15% frame paling senyap (jeda sebelum/
       sesudah/antar kata). Tidak perlu rekaman derau terpisah.
    2. Penguatan tiap titik waktu-frekuensi: G = 1 - oversub * derau / daya, dibatasi
       minimal `floor_db` agar suara tidak terdengar "berair".
    3. G dihaluskan (3 frekuensi x 5 frame) untuk mengurangi artefak.
    Pada rekaman studio derau hampir nol, sehingga G ~ 1 dan audio praktis tidak berubah.
    """
    if len(y) < DN_FFT * 2:
        return y
    S = librosa.stft(y, n_fft=DN_FFT, hop_length=DN_HOP)
    P = np.abs(S) ** 2
    e = P.sum(0)
    quiet = np.argsort(e)[:max(4, int(0.15 * P.shape[1]))]
    noise = P[:, quiet].mean(1, keepdims=True)
    gain = 1.0 - oversub * noise / (P + 1e-12)
    g_min = 10 ** (floor_db / 20)
    gain = np.clip(gain, g_min ** 2, 1.0) ** 0.5          # daya -> amplitudo
    gain = uniform_filter(gain, size=(3, 5), mode='nearest')
    out = librosa.istft(S * gain, hop_length=DN_HOP, length=len(y))
    return out.astype(np.float32)


def voice_bounds(y: np.ndarray, pad_sec: float = 0.15):
    """Awal/akhir bagian bersuara, dengan ambang RELATIF TERHADAP NOISE FLOOR.

    Versi 1 memakai ambang relatif terhadap puncak (trim top_db=35). Pada rekaman bising
    derau sering lebih keras dari ambang itu, sehingga jeda bising di awal/akhir ikut
    terbaca model dan memunculkan digit sisipan.
    """
    db = frame_db(y)
    if len(db) < 5:
        return 0, len(y)
    floor, peak = np.percentile(db, 10), np.percentile(db, 99)
    thr = max(floor + max(6.0, 0.25 * (peak - floor)), peak - 45.0)
    voiced = db > thr
    # buang ledakan pendek (< 50 ms), mis. klik tombol / ketukan meja
    voiced = np.convolve(voiced.astype(float), np.ones(5) / 5, mode='same') > 0.5
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        return 0, len(y)
    pad = int(pad_sec * SR)
    return max(0, idx[0] * 160 - pad), min(len(y), (idx[-1] + 1) * 160 + pad)


def preprocess(y: np.ndarray, enhance: bool = True, return_info: bool = False):
    """Rantai pra-pemrosesan yang SAMA untuk latih dan prediksi.

    high-pass 70 Hz -> penjernihan (opsional) -> potong jeda awal/akhir -> normalisasi RMS.
    Dengan return_info=True juga mengembalikan dict {offset (sampel), snr_db}.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size < 800:
        y = np.pad(y, (0, 800 - y.size))
    y = signal.sosfilt(_HP_SOS, y - y.mean()).astype(np.float32)
    snr = estimate_snr(y)
    if enhance:
        y = denoise(y)
    s, e = voice_bounds(y)
    y = y[s:e]
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    y = np.clip(y * (0.1 / rms), -1.0, 1.0).astype(np.float32)     # sekitar -20 dBFS
    return (y, {'offset': int(s), 'snr_db': snr}) if return_info else y


# =============================================================================
# 3. AUGMENTASI DERAU (KHUSUS PELATIHAN)
# =============================================================================
def colored_noise(n: int, rng, kind: str) -> np.ndarray:
    """Derau putih / pink / brown (spektrum 1/f^a)."""
    w = rng.standard_normal(n)
    if kind == 'white':
        return w
    alpha = {'pink': 0.5, 'brown': 1.0}[kind]
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    return np.fft.irfft(np.fft.rfft(w) / f ** alpha, n)


def hum_noise(n: int, rng) -> np.ndarray:
    """Dengung listrik/kipas: frekuensi dasar 50-120 Hz + harmonik + sedikit derau."""
    t = np.arange(n) / SR
    f0 = rng.uniform(50, 120)
    x = sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.3))
            for k in range(1, 6))
    return x + 0.3 * colored_noise(n, rng, 'pink') / 30


def synthetic_rir(rng, rt60=None) -> np.ndarray:
    """Respons impuls ruangan sederhana (gema): jalur langsung + ekor peluruhan eksponensial."""
    rt60 = rng.uniform(0.15, 0.7) if rt60 is None else rt60
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[:int(0.003 * SR)] = 0
    h = np.concatenate([[1.0], np.zeros(int(rng.uniform(0.001, 0.004) * SR))])
    h = np.pad(h, (0, max(0, n - len(h))))[:n]
    h = h + rng.uniform(0.1, 0.5) * tail / (np.abs(tail).max() + 1e-8)
    return h / np.sqrt(np.sum(h ** 2))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Campur derau ke ucapan pada SNR tertentu (daya dihitung pada bagian bersuara)."""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / max(len(noise), 1))))
    noise = noise[:len(clean)]
    p_s = np.mean(clean[np.abs(clean) > 0.02 * np.abs(clean).max()] ** 2) if clean.any() else 1.0
    p_n = np.mean(noise ** 2) + 1e-12
    return clean + noise * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))


class NoiseBank:
    """Sumber derau untuk augmentasi.

    babble_waves : daftar gelombang ucapan lain (dari penutur LATIH) -> derau "orang ngobrol".
    real_noises  : daftar gelombang derau nyata (opsional, mis. rekaman ruangan Anda sendiri).
    """

    def __init__(self, babble_waves=None, real_noises=None):
        self.babble = babble_waves or []
        self.real = [r for r in (real_noises or []) if len(r) > SR // 2]

    def sample(self, n: int, rng) -> np.ndarray:
        kinds = ['white', 'pink', 'brown', 'hum']
        if len(self.babble) >= 4:
            kinds += ['babble', 'babble']
        if self.real:
            kinds += ['real'] * 4                          # derau nyata diberi bobot lebih
        kind = kinds[rng.integers(len(kinds))]
        if kind in ('white', 'pink', 'brown'):
            return colored_noise(n, rng, kind)
        if kind == 'hum':
            return hum_noise(n, rng)
        if kind == 'babble':
            out = np.zeros(n)
            for _ in range(rng.integers(3, 7)):
                w = _as_float(self.babble[rng.integers(len(self.babble))])
                start = rng.integers(0, n)
                seg = np.tile(w, int(np.ceil(n / len(w))) + 1)[:n]
                out += np.roll(seg, start)
            return out
        w = _as_float(self.real[rng.integers(len(self.real))])
        if len(w) <= n:
            w = np.tile(w, int(np.ceil(n / len(w))) + 1)
        s = rng.integers(0, len(w) - n + 1)
        return w[s:s + n]


def _as_float(w):
    return w.astype(np.float32) / 32768.0 if w.dtype == np.int16 else w.astype(np.float32)


def augment(y: np.ndarray, rng, bank: NoiseBank | None = None, p_noise=0.8) -> np.ndarray:
    """Membuat versi 'rekaman mikrofon' dari ucapan studio (dipakai HANYA saat latih).

    Urutan: kecepatan bicara -> jeda hening awal/akhir -> gema ruangan -> karakter mikrofon
    (band-pass) -> derau aditif (SNR 0-30 dB) -> volume acak -> clipping.
    """
    y = _as_float(y)
    if rng.random() < 0.4:                                # kecepatan bicara 0,9x / 1,1x
        up, down = (10, 9) if rng.random() < 0.5 else (10, 11)
        y = signal.resample_poly(y, down, up).astype(np.float32)
    lead, tail = (rng.uniform(0, 0.8, 2) * SR).astype(int)  # pengguna diam sebelum/sesudah bicara
    y = np.pad(y, (lead, tail))
    if rng.random() < 0.35:
        y = signal.fftconvolve(y, synthetic_rir(rng))[:len(y)]
    if rng.random() < 0.5:                                # mikrofon murah / HP / headset
        lo, hi = rng.uniform(80, 400), rng.uniform(3400, 7600)
        y = signal.sosfilt(signal.butter(2, [lo, hi], 'bandpass', fs=SR, output='sos'), y)
    if bank is not None and rng.random() < p_noise:
        y = mix_at_snr(y, bank.sample(len(y), rng), rng.uniform(0, 30))
    elif rng.random() < 0.5:                              # tetap ada derau mikrofon tipis
        y = mix_at_snr(y, colored_noise(len(y), rng, 'white'), rng.uniform(30, 50))
    y = y / (np.abs(y).max() + 1e-8) * 10 ** (rng.uniform(-20, 0) / 20)
    if rng.random() < 0.05:                               # mikrofon terlalu dekat -> clipping
        y = np.clip(y * rng.uniform(2, 5), -1, 1)
    return y.astype(np.float32)


# =============================================================================
# 4. FITUR UNTUK MODEL CTC
# =============================================================================
N_MELS, N_FFT, HOP = 64, 400, 160          # jendela 25 ms, geser 10 ms


def ctc_features(y: np.ndarray) -> np.ndarray:
    """Log-mel 64 dimensi per frame + normalisasi rata-rata/varians per ucapan.

    Masukan = audio yang SUDAH melewati preprocess(). Hasil (jumlah_frame, 64).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmin=60, fmax=7600)
    feat = np.log(mel + 1e-6).T
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-5)
    return feat.astype(np.float32)


# =============================================================================
# 5. SEGMENTASI KATA + FITUR PER SEGMEN (untuk SVM / RF / XGBoost / KNN)
# =============================================================================
TOP_DB_GRID = (20, 25, 30, 35, 40)          # kandidat ambang hening (dB di bawah puncak)
MIN_GAP, MIN_DUR, PAD = 0.08, 0.10, 0.03    # detik


def segment_utterance(y: np.ndarray, top_db: float):
    """Memotong ucapan menjadi segmen kata berdasarkan jeda hening.

    Ambang = yang lebih tinggi dari (puncak - top_db) dan (noise floor + 8 dB), sehingga
    pada audio bising jeda antar-kata yang berisi derau tetap dianggap jeda.
    Jeda < MIN_GAP digabung, segmen < MIN_DUR dibuang. Mengembalikan [(awal, akhir)] sampel.
    """
    hop = 128
    db = frame_db(y, frame=512, hop=hop)
    if db.size == 0:
        return []
    peak, floor = db.max(), np.percentile(db, 10)
    thr = max(peak - top_db, floor + 8.0)
    voiced = np.concatenate([[False], db > thr, [False]])
    edges = np.flatnonzero(np.diff(voiced.astype(int)))
    raw = [(s * hop, min(len(y), e * hop)) for s, e in zip(edges[::2], edges[1::2])]
    merged = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) < MIN_GAP * SR:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    pad = int(PAD * SR)
    return [(max(0, s - pad), min(len(y), e + pad))
            for s, e in merged if (e - s) >= MIN_DUR * SR]


SEG_FEAT_DIM = 183


def segment_features(seg: np.ndarray) -> np.ndarray:
    """Vektor 183 fitur untuk satu segmen kata.

    78 = rata-rata & simpangan baku dari MFCC, delta, delta-delta (13 x 3 x 2)
   104 = bentuk MFCC sepanjang waktu, diringkas ke 8 titik waktu (13 x 8)
     1 = durasi segmen (detik)
    """
    if len(seg) < N_FFT:
        seg = np.pad(seg, (0, N_FFT - len(seg)))
    mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc, mode='nearest')
    d2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
    stack = np.vstack([mfcc, d1, d2])
    src = np.linspace(0, 1, mfcc.shape[1])
    shape = np.array([np.interp(np.linspace(0, 1, 8), src, row) for row in mfcc]).ravel()
    return np.concatenate([stack.mean(1), stack.std(1), shape,
                           [len(seg) / SR]]).astype(np.float32)


def segments_for_all_db(y: np.ndarray) -> dict:
    """Fitur segmen untuk setiap ambang di TOP_DB_GRID (y = hasil preprocess)."""
    out = {}
    for db in TOP_DB_GRID:
        iv = segment_utterance(y, db)
        out[db] = (np.stack([segment_features(y[s:e]) for s, e in iv])
                   if iv else np.zeros((0, SEG_FEAT_DIM), np.float32))
    return out


def extract_all(source, enhance: bool = True) -> dict:
    """Satu kali baca audio -> semua fitur yang dibutuhkan semua metode (untuk cache).

    `source` boleh path file atau gelombang numpy (mis. versi bising untuk evaluasi).
    """
    y0 = load_audio(source) if isinstance(source, (str, bytes, bytearray)) else _as_float(source)
    y = preprocess(y0, enhance=enhance)
    return {'ctc': ctc_features(y).astype(np.float16), 'segs': segments_for_all_db(y),
            'duration': len(y0) / SR}


# =============================================================================
# 6. MODEL CNN-BiLSTM-CTC
# =============================================================================
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:          # dashboard tetap jalan tanpa PyTorch (metode CTC dinonaktifkan)
    TORCH_OK = False

BLANK = 0                    # indeks 0 = blank CTC; token ke-i -> indeks i+1
N_OUT = len(TOKENS) + 1

if TORCH_OK:
    class CTCNet(nn.Module):
        """CNN 2D (pola waktu-frekuensi, tahan pergeseran nada/karakter mikrofon)
        -> BiLSTM 2 lapis (konteks urutan) -> peluang tiap token per frame.
        Waktu diperpendek 2x, frekuensi 4x."""

        def __init__(self, n_in=N_MELS, hidden=192, n_out=N_OUT, ch=32, dropout=0.3):
            super().__init__()

            def block(cin, cout):
                return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU()]

            self.conv = nn.Sequential(
                *block(1, ch), *block(ch, ch), nn.MaxPool2d((2, 2)),
                *block(ch, 2 * ch), *block(2 * ch, 2 * ch), nn.MaxPool2d((1, 2)),
                nn.Dropout2d(0.1))
            self.proj = nn.Sequential(nn.Linear(2 * ch * (n_in // 4), 256), nn.GELU(),
                                      nn.Dropout(dropout))
            self.rnn = nn.LSTM(256, hidden, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=dropout)
            self.out = nn.Linear(2 * hidden, n_out)

        def forward(self, x):                      # x: (batch, frame, n_mels)
            h = self.conv(x.unsqueeze(1))          # (batch, ch, frame/2, n_mels/4)
            b, c, t, f = h.shape
            h = self.proj(h.permute(0, 2, 1, 3).reshape(b, t, c * f))
            h, _ = self.rnn(h)
            return self.out(h).log_softmax(-1)     # (batch, frame/2, N_OUT)

    def out_lengths(lengths):
        """Panjang keluaran model setelah MaxPool waktu 2x."""
        return lengths // 2


def ctc_greedy_decode(logp: np.ndarray):
    """Decoding greedy: ambil token terbaik per frame, gabungkan pengulangan, buang blank.

    Mengembalikan (token, keyakinan per token, frame tengah per token).
    """
    prob = np.exp(logp)
    best = prob.argmax(1)
    tokens, confs, frames = [], [], []
    prev, run = BLANK, []
    for t, k in enumerate(np.append(best, BLANK)):
        if k != prev and prev != BLANK and run:
            tokens.append(TOKENS[prev - 1])
            confs.append(float(prob[run, prev].max()))
            frames.append(int(np.mean(run)))
            run = []
        if k != BLANK and t < len(best):
            run = run + [t] if k == prev else [t]
        prev = k
    return tokens, confs, frames


# =============================================================================
# 7. METRIK URUTAN
# =============================================================================
def align(ref, hyp):
    """Alignment Levenshtein -> daftar pasangan (ref, hyp); None = sisipan/hapusan."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            pairs.append((ref[i - 1], None)); i -= 1
        else:
            pairs.append((None, hyp[j - 1])); j -= 1
    return pairs[::-1]


def error_counts(ref, hyp):
    """(substitusi, hapusan, sisipan) antara urutan referensi dan hipotesis."""
    S = D = I = 0
    for r, h in align(ref, hyp):
        if r is None:
            I += 1
        elif h is None:
            D += 1
        elif r != h:
            S += 1
    return S, D, I


# =============================================================================
# 8. RECOGNIZER — dipakai notebook (uji kecepatan) dan dashboard
# =============================================================================
METHOD_INFO = {
    'CNN-BiLSTM-CTC': 'Deep learning end-to-end: CNN 2D membaca spektrogram, BiLSTM membaca '
                      'urutan, CTC langsung menghasilkan urutan digit tanpa memotong audio. '
                      'Dilatih dengan augmentasi derau/gema/mikrofon agar tahan rekaman bising.',
    'SVM': 'Audio dipotong per kata pada jeda hening; tiap potongan diklasifikasi '
           'Support Vector Machine (kernel RBF).',
    'Random Forest': 'Audio dipotong per kata; tiap potongan diklasifikasi oleh '
                     'ratusan pohon keputusan yang memberi suara mayoritas.',
    'XGBoost': 'Audio dipotong per kata; tiap potongan diklasifikasi gradient '
               'boosting (pohon dibangun bertahap memperbaiki kesalahan sebelumnya).',
    'KNN': 'Audio dipotong per kata; tiap potongan diberi label mayoritas dari '
           'k contoh latih yang paling mirip.',
}
SEGMENT_METHODS = ['SVM', 'Random Forest', 'XGBoost', 'KNN']
MODEL_FILES = {'SVM': 'svm.joblib', 'Random Forest': 'random_forest.joblib',
               'XGBoost': 'xgboost.joblib', 'KNN': 'knn.joblib',
               'CNN-BiLSTM-CTC': 'ctc.pt'}
ENH_SNR_DB = 25.0            # perkiraan SNR di atas ini: rekaman dianggap jernih
SNR_WARN_DB = 15.0           # perkiraan SNR di bawah ini: tampilkan peringatan
ENHANCE_MODES = {            # pilihan penjernihan untuk dashboard (kunci pertama = bawaan)
    'auto': 'Otomatis — coba asli & dijernihkan, pakai yang paling yakin',
    'on': 'Selalu dijernihkan',
    'off': 'Mati (tanpa penjernihan)',
}
SNR_BAD_DB = 8.0             # di bawah ini: sarankan rekam ulang (estimator cenderung +2-3 dB lebih tinggi)


def quality_message(snr_db: float, dur_sec: float) -> str | None:
    """Pesan kualitas rekaman untuk pengguna (None = aman)."""
    if dur_sec < 0.3:
        return 'Suara terlalu pendek atau tidak terdengar. Coba rekam ulang lebih dekat ke mikrofon.'
    if snr_db < SNR_BAD_DB:
        return (f'Rekaman sangat bising (SNR ~{snr_db:.0f} dB). Hasil kemungkinan salah; '
                'pindah ke tempat lebih tenang atau dekatkan mikrofon.')
    if snr_db < SNR_WARN_DB:
        return f'Rekaman cukup bising (SNR ~{snr_db:.0f} dB); suara sudah dijernihkan, periksa hasilnya.'
    return None


class Recognizer:
    """Memuat model dari folder artefak dan mengenali urutan digit dari audio."""

    def __init__(self, artifact_dir: str):
        self.dir = artifact_dir
        with open(os.path.join(artifact_dir, 'config.json')) as f:
            self.config = json.load(f)
        self.top_db = self.config['segment_top_db']
        self.enhance = self.config.get('enhance', True)          # kondisi saat model dilatih
        self.default_mode = self.config.get('enhance_mode', 'auto')
        self._models = {}

    def available(self):
        """Metode yang file modelnya ada (CTC hanya bila PyTorch terpasang)."""
        out = []
        for name, fn in MODEL_FILES.items():
            if os.path.exists(os.path.join(self.dir, 'models', fn)):
                if name == 'CNN-BiLSTM-CTC' and not TORCH_OK:
                    continue
                out.append(name)
        return out

    def _model(self, name):
        if name not in self._models:
            path = os.path.join(self.dir, 'models', MODEL_FILES[name])
            if name == 'CNN-BiLSTM-CTC':
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                net = CTCNet(**ckpt['arch'])
                net.load_state_dict(ckpt['state_dict'])
                self._models[name] = net.eval()
            else:
                self._models[name] = joblib.load(path)
        return self._models[name]

    def _mode(self, enhance) -> str:
        """Menerima True/False (kompatibel versi lama) atau kunci ENHANCE_MODES."""
        if enhance is None:
            return self.default_mode
        if isinstance(enhance, bool):
            return 'on' if enhance else 'off'
        if enhance not in ENHANCE_MODES:
            raise ValueError(f'Mode penjernihan tidak dikenal: {enhance}')
        return enhance

    def _recognize_once(self, y: np.ndarray, method: str, enhance: bool) -> dict:
        yc, info = preprocess(y, enhance=enhance, return_info=True)
        off = info['offset'] / SR
        if method == 'CNN-BiLSTM-CTC':
            feat = ctc_features(yc)
            with torch.no_grad():
                logp = self._model(method)(torch.from_numpy(feat)[None])[0].numpy()
            tokens, confs, frames = ctc_greedy_decode(logp)
            times = [off + f * 2 * HOP / SR for f in frames]
            segments = None
        else:
            iv = segment_utterance(yc, self.top_db)
            tokens, confs, times = [], [], []
            if iv:
                X = np.stack([segment_features(yc[s:e]) for s, e in iv])
                model = self._model(method)
                proba = model.predict_proba(X)
                idx = proba.argmax(1)
                tokens = [TOKENS[int(model.classes_[i])] for i in idx]
                confs = proba.max(1).astype(float).tolist()
                times = [off + (s + e) / 2 / SR for s, e in iv]
            segments = [(off + s / SR, off + e / SR) for s, e in iv]
        return {'tokens': tokens, 'confidence': confs, 'times': times, 'segments': segments,
                'audio': yc, 'snr_db': info['snr_db'], 'enhanced': enhance}

    def recognize(self, y: np.ndarray, method: str, enhance=None) -> dict:
        """Mengenali satu ucapan.

        enhance: 'auto' | 'on' | 'off' (atau True/False). 'auto' mengenali audio yang
        dijernihkan DAN audio asli, lalu memakai hasil dengan keyakinan rata-rata tertinggi
        (seri -> versi yang sama dengan kondisi latih). Pada uji derau, 'auto' memberi WER
        terendah atau setara dibanding 'on'/'off' di semua tingkat SNR.

        Mengembalikan token, angka, kalimat, keyakinan, waktu (detik, relatif ke audio
        masukan), segmen, perkiraan SNR, pesan kualitas, `variant` ('dijernihkan'/'asli'),
        dan `audio` = audio yang benar-benar dibaca model (`audio_clean` = alias lama).
        """
        mode = self._mode(enhance)
        self._model(method)                     # muat model dulu agar waktu = waktu mengenali saja
        t0 = time.perf_counter()
        order = [self.enhance, not self.enhance] if mode == 'auto' else [mode == 'on']
        cands = [self._recognize_once(y, method, e) for e in order]
        score = lambda c: float(np.mean(c['confidence'])) if c['confidence'] else -1.0
        best = max(cands, key=score)            # max() memilih yang pertama bila seri
        tokens, yc = best['tokens'], best['audio']
        return {'method': method, 'mode': mode, 'tokens': tokens, 'digits': ''.join(to_value(tokens)),
                'words': to_words(tokens), 'confidence': best['confidence'], 'times': best['times'],
                'segments': best['segments'], 'latency_ms': (time.perf_counter() - t0) * 1000,
                'snr_db': best['snr_db'], 'warning': quality_message(best['snr_db'], len(yc) / SR),
                'variant': 'dijernihkan' if best['enhanced'] else 'asli',
                'audio': yc, 'audio_clean': yc}
