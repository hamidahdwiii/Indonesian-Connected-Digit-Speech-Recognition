"""
cdsr_core.py — Modul inti Indonesian Connected Digit Speech Recognition (CDSR).

Satu sumber kode untuk NOTEBOOK (pelatihan & evaluasi) dan DASHBOARD (inferensi),
sehingga cara audio diproses saat latih dan saat dipakai dijamin sama persis.

Isi modul:
  1. Kosakata & parsing transkrip dari nama file   -> parse_filename()
  2. Pemuatan audio                                 -> load_audio()
  3. Fitur untuk model CTC (log-mel per frame)      -> ctc_features()
  4. Segmentasi kata + fitur per segmen             -> segment_utterance(), segment_features()
  5. Model CNN-BiLSTM-CTC (PyTorch) + decoding      -> CTCNet, ctc_greedy_decode()
  6. Metrik urutan (WER/SER, alignment)             -> align(), error_counts()
  7. Recognizer: memuat artefak & mengenali audio   -> Recognizer
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
# 2. PEMUATAN AUDIO
# =============================================================================
def load_audio(source) -> np.ndarray:
    """Memuat audio (path, bytes, atau file-like) -> mono 16 kHz, puncak dinormalkan ke 1."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, _ = librosa.load(source, sr=SR, mono=True)
    peak = np.max(np.abs(y)) if y.size else 0.0
    return (y / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


# =============================================================================
# 3. FITUR UNTUK MODEL CTC
# =============================================================================
N_MELS, N_FFT, HOP = 40, 400, 160          # jendela 25 ms, geser 10 ms


def ctc_features(y: np.ndarray) -> np.ndarray:
    """Log-mel 40 dimensi per frame + normalisasi rata-rata/varians per ucapan.

    Hasil berbentuk (jumlah_frame, 40): urutan waktu DIPERTAHANKAN, inilah yang
    memungkinkan model mengenali beberapa digit dalam satu file.
    """
    y, _ = librosa.effects.trim(y, top_db=35)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS)
    feat = np.log(mel + 1e-6).T
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-5)
    return feat.astype(np.float32)


# =============================================================================
# 4. SEGMENTASI KATA + FITUR PER SEGMEN (untuk SVM / RF / XGBoost / KNN)
# =============================================================================
TOP_DB_GRID = (20, 25, 30, 35, 40)          # kandidat ambang hening (dB di bawah puncak)
MIN_GAP, MIN_DUR, PAD = 0.08, 0.10, 0.03    # detik


def segment_utterance(y: np.ndarray, top_db: float):
    """Memotong ucapan menjadi segmen kata berdasarkan jeda hening.

    Jeda < MIN_GAP digabung (mis. penutupan konsonan 'p' pada 'empat'),
    segmen < MIN_DUR dibuang (bunyi klik/napas). Mengembalikan [(awal, akhir)] sampel.
    """
    raw = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=128)
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


def extract_all(path: str) -> dict:
    """Satu kali baca audio -> semua fitur yang dibutuhkan semua metode (untuk cache)."""
    y = load_audio(path)
    segs = {}
    for db in TOP_DB_GRID:
        iv = segment_utterance(y, db)
        feats = (np.stack([segment_features(y[s:e]) for s, e in iv])
                 if iv else np.zeros((0, SEG_FEAT_DIM), np.float32))
        segs[db] = feats
    return {'ctc': ctc_features(y).astype(np.float16), 'segs': segs,
            'duration': len(y) / SR}


# =============================================================================
# 5. MODEL CNN-BiLSTM-CTC
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
        """2 lapis konvolusi (pola akustik lokal) -> 2 lapis BiLSTM (konteks urutan)
        -> peluang tiap token per frame. Pooling memperpendek urutan 2x."""

        def __init__(self, n_in=N_MELS, hidden=128, n_out=N_OUT):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_in, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(2))
            self.rnn = nn.LSTM(128, hidden, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=0.3)
            self.out = nn.Linear(2 * hidden, n_out)

        def forward(self, x):                      # x: (batch, frame, 40)
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.rnn(h)
            return self.out(h).log_softmax(-1)     # (batch, frame/2, N_OUT)

    def out_lengths(lengths):
        """Panjang keluaran model setelah MaxPool(2)."""
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
# 6. METRIK URUTAN
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
# 7. RECOGNIZER — dipakai notebook (uji kecepatan) dan dashboard
# =============================================================================
METHOD_INFO = {
    'CNN-BiLSTM-CTC': 'Jaringan saraf end-to-end: membaca seluruh rekaman dan langsung '
                      'menghasilkan urutan digit tanpa memotong audio.',
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


class Recognizer:
    """Memuat model dari folder artefak dan mengenali urutan digit dari audio."""

    def __init__(self, artifact_dir: str):
        self.dir = artifact_dir
        with open(os.path.join(artifact_dir, 'config.json')) as f:
            self.config = json.load(f)
        self.top_db = self.config['segment_top_db']
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

    def recognize(self, y: np.ndarray, method: str) -> dict:
        """Mengenali satu ucapan. Mengembalikan token, angka, kalimat, keyakinan, waktu."""
        self._model(method)                     # muat model dulu agar waktu = waktu mengenali saja
        t0 = time.perf_counter()
        if method == 'CNN-BiLSTM-CTC':
            feat = ctc_features(y)
            with torch.no_grad():
                logp = self._model(method)(torch.from_numpy(feat)[None])[0].numpy()
            tokens, confs, frames = ctc_greedy_decode(logp)
            trim_start = librosa.effects.trim(y, top_db=35)[1][0]
            times = [(trim_start + f * 2 * HOP) / SR for f in frames]
            segments = None
        else:
            iv = segment_utterance(y, self.top_db)
            tokens, confs, times = [], [], []
            if iv:
                X = np.stack([segment_features(y[s:e]) for s, e in iv])
                model = self._model(method)
                proba = model.predict_proba(X)
                idx = proba.argmax(1)
                tokens = [TOKENS[int(model.classes_[i])] for i in idx]
                confs = proba.max(1).astype(float).tolist()
                times = [(s + e) / 2 / SR for s, e in iv]
            segments = [(s / SR, e / SR) for s, e in iv]
        return {'method': method, 'tokens': tokens, 'digits': ''.join(to_value(tokens)),
                'words': to_words(tokens), 'confidence': confs, 'times': times,
                'segments': segments, 'latency_ms': (time.perf_counter() - t0) * 1000}
