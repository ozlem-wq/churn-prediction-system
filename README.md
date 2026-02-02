# Churn Prediction System

Müşteri kaybını (churn) tahmin eden uçtan uca bir makine öğrenmesi sistemi.

## Özellikler

- **ML Pipeline**: Veri yükleme, ön işleme, feature engineering, model eğitimi
- **REST API**: FastAPI ile tahmin endpoint'leri
- **Dashboard**: Streamlit ile interaktif görselleştirme
- **MLflow**: Experiment tracking ve model registry
- **Docker**: Konteynerize edilmiş servisler
- **Kubernetes**: Production-ready deployment

## Hızlı Başlangıç

### Gereksinimler

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15 (Docker ile sağlanır)

### Kurulum

```bash
# 1. Repository'yi klonlayın
cd churn-prediction-system

# 2. Virtual environment oluşturun
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya: venv\Scripts\activate  # Windows

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Environment dosyasını oluşturun
cp .env.example .env

# 5. Docker ile servisleri başlatın
docker-compose up -d
```

### Veri Seti

Telco Customer Churn veri setini indirin:
- [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

İndirilen CSV dosyasını `data/raw/` klasörüne koyun:
```bash
mv WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/
```

## Kullanım

### 1. Jupyter Notebooks ile Keşif

```bash
jupyter notebook notebooks/
```

Notebook sırası:
1. `01_data_exploration.ipynb` - Keşifsel veri analizi
2. `02_feature_engineering.ipynb` - Özellik mühendisliği
3. `03_model_training.ipynb` - Model eğitimi

### 2. API'yi Çalıştırma

```bash
# Geliştirme modu
uvicorn src.api.main:app --reload --port 8000

# veya Docker ile
docker-compose up api
```

API Dokümantasyonu: http://localhost:8000/docs

### 3. Dashboard'u Çalıştırma

```bash
streamlit run streamlit_app/app.py

# veya Docker ile
docker-compose up streamlit
```

Dashboard: http://localhost:8501

### 4. Tahmin Yapma

```bash
# cURL ile tek tahmin
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "NEW-001",
    "gender": "Male",
    "tenure_months": 6,
    "monthly_charges": 75.50,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'
```

## Proje Yapısı

```
churn-prediction-system/
├── data/
│   ├── raw/                 # Ham veri
│   └── processed/           # İşlenmiş veri
├── notebooks/               # Jupyter notebooks
├── sql/                     # Database şemaları
├── src/
│   ├── api/                 # FastAPI uygulaması
│   ├── data/                # Veri yükleme/işleme
│   ├── features/            # Feature engineering
│   └── models/              # Model eğitimi/tahmin
├── models/                  # Eğitilmiş modeller
├── tests/                   # Test dosyaları
├── streamlit_app/           # Dashboard
├── docker/                  # Dockerfile'lar
├── k8s/                     # Kubernetes manifests
└── .github/workflows/       # CI/CD pipelines
```

## API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/health` | GET | Sağlık kontrolü |
| `/api/v1/customers` | GET | Müşteri listesi |
| `/api/v1/customers/{id}` | GET | Müşteri detayı |
| `/api/v1/predict` | POST | Tek tahmin |
| `/api/v1/predict/batch` | POST | Toplu tahmin |
| `/api/v1/model/info` | GET | Model bilgisi |
| `/api/v1/stats` | GET | İstatistikler |

## Testleri Çalıştırma

```bash
# Tüm testler
pytest tests/ -v

# Coverage ile
pytest tests/ -v --cov=src --cov-report=html
```

## Docker Komutları

```bash
# Tüm servisleri başlat
docker-compose up -d

# Logları izle
docker-compose logs -f

# Servisleri durdur
docker-compose down

# Veritabanını sıfırla
docker-compose down -v
docker-compose up -d
```

## Kubernetes Deployment

```bash
# Namespace oluştur
kubectl apply -f k8s/namespace.yaml

# Config ve secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# PostgreSQL
kubectl apply -f k8s/postgres/

# API
kubectl apply -f k8s/api/

# Streamlit
kubectl apply -f k8s/streamlit/

# Ingress
kubectl apply -f k8s/ingress.yaml
```

## Model Performansı

| Metrik | Hedef | Sonuç |
|--------|-------|-------|
| Accuracy | > 80% | ~82% |
| Precision | > 75% | ~78% |
| Recall | > 70% | ~75% |
| F1-Score | > 72% | ~76% |
| ROC-AUC | > 0.80 | ~0.85 |

## Teknolojiler

- **Python 3.11**: Ana programlama dili
- **FastAPI**: Async REST API
- **Streamlit**: Dashboard
- **PostgreSQL**: Veritabanı
- **SQLAlchemy**: ORM
- **Pandas/NumPy**: Veri işleme
- **Scikit-learn**: ML algoritmaları
- **XGBoost**: Gradient boosting
- **MLflow**: Experiment tracking
- **Docker**: Konteynerizasyon
- **Kubernetes**: Orkestrasyon

## Lisans

MIT License

## İletişim

Sorularınız için issue açabilirsiniz.
