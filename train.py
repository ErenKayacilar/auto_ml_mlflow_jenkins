import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow import MlflowClient
import os

# --------------------------------------------------------------
# 🔧 MLflow bağlantı ayarı (otomatik kontrol)
# Önce environment değişkeninden oku, yoksa local dosyaya kaydet
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Yorum Onaylama Deneyi")

REGISTERED_MODEL_NAME = "YorumOnayModeli"
PRODUCTION_ALIAS = "Production"

# --------------------------------------------------------------
def create_dummy_data(rows=200):
    """Basit sahte veri oluşturur."""
    data = {
        'yorum': [
            "bu ürün harika çok beğendim", "kesinlikle tavsiye etmiyorum berbat",
            "fiyatına göre iyi bir performans", "kargo çok geç geldi",
            "mükemmel kalite ve tasarım", "bir daha asla almam",
            "eh işte idare eder", "beklentimi karşılamadı",
            "gerçekten çok başarılı bir ürün", "malzemesi çok kalitesiz"
        ] * (rows // 10),
        'puan': [float(random.randint(1, 5)) for _ in range(rows)],
        'onaylandi': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * (rows // 10)
    }
    return pd.DataFrame(data)

# --------------------------------------------------------------
def main():
    print("Veri oluşturuluyor...")
    df = create_dummy_data(200)

    X = df[['yorum', 'puan']]
    y = df['onaylandi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'yorum'),
            ('numeric', StandardScaler(), ['puan'])
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=5.0, solver='liblinear'))
    ])

    # --------------------------------------------------------------
    mlflow.autolog(log_models=False)

    with mlflow.start_run(run_name="LogisticRegression_C5") as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        print("Model eğitiliyor...")
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_train.head(3)

        print("Model MLflow'a kaydediliyor...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example
        )

        # ----------------------------------------------------------
        try:
            client = MlflowClient(tracking_uri=MLFLOW_URI)
            latest_version_info = client.get_latest_versions(REGISTERED_MODEL_NAME)[0]
            version = latest_version_info.version

            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias=PRODUCTION_ALIAS,
                version=version
            )
            print(f"🚀 '{REGISTERED_MODEL_NAME}' modelinin {version}. versiyonu '{PRODUCTION_ALIAS}' alias'ına atandı.")
        except Exception as e:
            print(f"⚠️ Alias eklenemedi (MLflow server kapalı olabilir): {e}")

        print("Run tamamlandı.")
        print(f"📊 Run klasörü: {os.path.abspath('mlruns')}")

# --------------------------------------------------------------
if __name__ == "__main__":
    main()
