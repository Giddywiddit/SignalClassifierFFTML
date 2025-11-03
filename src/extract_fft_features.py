import numpy as np

def fft_features(batch, Fs=8000, top_k=20):
    N, L = batch.shape
    Xf = np.fft.fft(batch, axis=1)
    mag = np.abs(Xf[:, :L//2])
    idx = np.argpartition(mag, -top_k, axis=1)[:, -top_k:]
    rows = np.arange(N)[:, None]
    feats = mag[rows, idx]
    feats.sort(axis=1)
    feats = feats[:, ::-1]
    return feats.astype(np.float32)

if __name__ == "__main__":
    X = np.load("data/signals.npy")
    F = fft_features(X, Fs=8000, top_k=20)
    np.save("data/features.npy", F)
    print("Saved data/features.npy", F.shape)
