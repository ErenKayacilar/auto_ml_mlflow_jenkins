from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

model_name = "YorumOnayModeli"
version = 1
alias = "Production"

client.set_registered_model_alias(
    name=model_name,
    alias=alias,
    version=version
)

print(f"✅ '{model_name}' modelinin {version}. versiyonu için '{alias}' alias'ı oluşturuldu.")
