import numpy as np

def synth_tone(freq, Fs=8000, T=1.0, amp=15, phase=None, noise_std=0.5):
    t = np.linspace(0, T, int(Fs*T), endpoint=False)
    if phase is None:
        phase = np.random.uniform(0, 2*np.pi)
    x = amp * np.sin(2*np.pi*freq*t + phase)
    if noise_std > 0:
        x += np.random.normal(0, noise_std, len(t))
    return x

def make_dataset(classes=(770, 785), samples_per_class=300, Fs=8000, T=1.0):
    X, y = [], []
    for idx, f in enumerate(classes):
        for _ in range(samples_per_class):
            x = synth_tone(f, Fs=Fs, T=T, amp=15, noise_std=0.5)
            X.append(x.astype(np.float32))
            y.append(idx)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y

if __name__ == "__main__":
    X, y = make_dataset()
    np.save("data/signals.npy", X)
    np.save("data/labels.npy", y)
    print("Saved data/signals.npy and data/labels.npy", X.shape, y.shape)
