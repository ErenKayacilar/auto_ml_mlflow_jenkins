import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

REGISTERED_MODEL_NAME = "YorumOnayModeli"
MODEL_ALIAS = "Production" 

def load_model_by_alias():
    """MLflow'dan belirtilen takma ada sahip modeli yükler."""
    
    # MLflow sunucusuna bağlanıyoruz
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    
    try:
        # "Production" takma adına sahip en son versiyonu buluyoruz
        latest_version_info = client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME, 
            alias=MODEL_ALIAS
        )
        
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version_info.version}"
        
        print(f"'{MODEL_ALIAS}' takma adına sahip modelin {latest_version_info.version}. versiyonu yükleniyor.")
        print(f"Model URI: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        print("Model başarıyla yüklendi.")
        return model

    except Exception as e:
        print(f"HATA: Model yüklenemedi. '{MODEL_ALIAS}' takma adına sahip bir model var mı?")
        print(f"Hata detayı: {e}")
        return None

def predict(model, data):
    """Yüklenen model ile tahmin yapar."""
    if model is None:
        print("Tahmin yapılamıyor, model yüklenemedi.")
        return
    
    df = pd.DataFrame(data)

    # 🔧 MLflow schema uyumu için "puan" kolonunu float'a çeviriyoruz
    if "puan" in df.columns:
        df["puan"] = df["puan"].astype(float)

    prediction = model.predict(df)
    return prediction

if __name__ == "__main__":
    # Modelimizi yüklüyoruz
    production_model = load_model_by_alias()

    # Tahmin için örnek veri
    sample_data = {
        "yorum": ["bu harika bir telefon çok sevdim", "kamera çok kötü çıktı"],
        "puan": [5, 1]
    }
    
    # Tahmin yapıyoruz
    results = predict(production_model, sample_data)
    
    if results is not None:
        for i, res in enumerate(results):
            onay_durumu = "Onaylandı" if res == 1 else "Reddedildi"
            print(f"'{sample_data['yorum'][i]}' yorumu için tahmin: {onay_durumu}")