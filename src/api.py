from flask import Flask, request, jsonify
import numpy as np
from predict import predict_signal

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if "signal" not in data:
        return jsonify({"error": "missing 'signal' array"}), 400
    try:
        arr = np.array(data["signal"], dtype=float)
    except Exception as e:
        return jsonify({"error": f"invalid signal: {e}"}), 400
    pred = predict_signal(arr)
    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
