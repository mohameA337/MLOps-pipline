import onnxruntime as ort
import numpy as np
import os

MODEL_PATH = "model/titanic_model.onnx"
INPUT_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

class TitanicONNXPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.session = ort.InferenceSession(model_path)

    def predict(self, input_dict):
        try:
            input_data = np.array([[input_dict[feature] for feature in INPUT_FEATURES]], dtype=np.float32)
            result = self.session.run(None, {"input": input_data})[0]
            predicted_class = int(np.argmax(result))
            confidence = float(np.max(result))
            return {
                "prediction": predicted_class,
                "confidence": round(confidence, 4)
            }
        except Exception as e:
            raise RuntimeError(f"Inference error: {str(e)}")
