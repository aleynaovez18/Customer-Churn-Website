from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel
import xgboost as xgb

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 1. MODEL VE ENCODER YÜKLEME
try:
    with open('churn_model_final.pkl', 'rb') as file:
        saved_data = pickle.load(file)
        model = saved_data["model"]
        encoder = saved_data.get("encoder") or saved_data.get("encoders")
    print("Sistem başarıyla ayağa kalktı!")
except Exception as e:
    print(f"Yükleme Hatası: {e}")


# 2. VERİ MODELİ (Genişletilmiş sürüm)
class ChurnFeatures(BaseModel):
    Contract: str
    PaperlessBilling: str
    MultipleLines: str
    InternetService: str
    PaymentMethod: str
    SeniorCitizen: float
    tenure: float
    MonthlyCharges: float
    TotalServicesCount: float
    IsFamily: float


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(features: ChurnFeatures):
    input_df = pd.DataFrame([features.model_dump()])

    # Preprocessing
    input_encoded = encoder.transform(input_df)
    input_final = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())

    # Tahmin
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    # Akıllı Öneri Motoru
    recommendation = ""
    if prediction == 1:
        result_text = "Ayrılma Riski Yüksek"
        if features.Contract == "Month-to-month":
            recommendation = "Müşteri aylık sözleşmede. Sadakati artırmak için 1 yıllık avantajlı paket teklif edilmeli."
        elif features.TotalServicesCount < 2:
            recommendation = "Müşteri çok az ek hizmet kullanıyor. Güvenlik paketi veya TV aboneliği hediye edilerek bağ kurulabilir."
        else:
            recommendation = "Yüksek fatura itirazı olabilir. Kişiye özel %15 sadakat indirimi tanımlanması önerilir."
    else:
        result_text = "Müşteri Sadık"
        if probability < 0.20:
            recommendation = "Mükemmel skor! Bu müşteriye 'VIP Destek' statüsü verilerek yeni ürünlerin test süreci sunulabilir."
        else:
            recommendation = "Sadakat stabil ancak rakiplerin Fiber tekliflerine karşı takipte kalınmalı."


    return {
        "prediction": result_text,
        "probability": f"%{probability * 100:.2f}",
        "recommendation": recommendation,
        "is_churn": int(prediction)
    }