import numpy as np
import matplotlib.pyplot as plt
import os

def plot_example_ffts():
    Fs = 8000
    X = np.load("data/signals.npy")
    y = np.load("data/labels.npy")

    cls0 = X[y == 0][0]
    cls1 = X[y == 1][0]

    def mag_fft(x):
        L = len(x)
        mag = np.abs(np.fft.fft(x))[:L//2]
        return mag

    m0 = mag_fft(cls0)
    m1 = mag_fft(cls1)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(m0, label="Class 0 (770 Hz)")
    plt.plot(m1, label="Class 1 (785 Hz)")
    plt.title("FFT Magnitude (Examples)")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fft_examples.png", dpi=180)
    print("Saved plots/fft_examples.png")

if __name__ == "__main__":
    plot_example_ffts()
