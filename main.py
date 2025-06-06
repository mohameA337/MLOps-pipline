from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from inference import TitanicONNXPredictor

app = FastAPI(title="Titanic ONNX Prediction API")

predictor = TitanicONNXPredictor()

class TitanicInput(BaseModel):
    Pclass: int = Field(..., example=3)
    Sex: int = Field(..., example=1, description="0 = female, 1 = male")
    Age: float = Field(..., example=22.0)
    SibSp: int = Field(..., example=1)
    Parch: int = Field(..., example=0)
    Fare: float = Field(..., example=7.25)
    Embarked: int = Field(..., example=0, description="0 = C, 1 = Q, 2 = S")

@app.post("/predict")
def predict(data: TitanicInput):
    """
    Accepts Titanic passenger info as JSON and returns survival prediction.
    """
    try:
        result = predictor.predict(data.dict())
        return {
            "input": data.dict(),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "meaning": "1 = Survived, 0 = Did not survive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
