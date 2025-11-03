import numpy as np
import joblib
from extract_fft_features import fft_features

def predict_signal(x, Fs=8000, top_k=20):
    pack = joblib.load("models/fft_clf.joblib")
    scaler = pack["scaler"]
    model = pack["model"]
    feats = fft_features(np.expand_dims(x, 0), Fs=Fs, top_k=top_k)
    feats = scaler.transform(feats)
    pred = model.predict(feats)[0]
    return int(pred)

if __name__ == "__main__":
    t = np.linspace(0, 1, 8000, endpoint=False)
    x = 15*np.sin(2*np.pi*770*t) + np.random.normal(0,0.5,len(t))
    print("Predicted class:", predict_signal(x))
