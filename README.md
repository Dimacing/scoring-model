# Credit Scoring Model (PD)

Учебный проект по автоматизации ML-пайплайна для скоринговой модели (UCI Credit Card).

## Быстрый старт (Windows 11)

```powershell
git clone <this-repo>
cd credit-scoring-model
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
dvc init
dvc repro
```

## Запуск API

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## MLflow UI

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000
```
