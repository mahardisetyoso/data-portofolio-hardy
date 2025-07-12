# 🚕 Pickup Zone Optimization for Ride-Hailing

## 🎯 Project Description
This project aims to identify and optimize the most strategic pickup zones for ride-hailing services using spatial data analysis. Data is collected from public sources and simulated trip logs, stored centrally in Supabase (PostgreSQL), and visualized through an interactive Streamlit dashboard.

## 💡 Business Objectives
- ⏱️ Reduce passenger wait time
- ⚡ Improve driver efficiency
- 📍 Identify strategic pickup zones based on temporal and spatial patterns

## 🛠️ Tools & Technologies
- Python (pandas, geopandas, folium, plotly, SQLAlchemy)
- Streamlit for the interactive dashboard
- Supabase PostgreSQL (connected via session pooler port `5432`)
- Docker & Docker Compose for local containerized development
- `.env` + python-dotenv for secure environment variable handling

## 📦 Key Features
- ✅ Interactive pickup zone map visualization
- ✅ Query spatial data from Supabase using SQLAlchemy
- ✅ Secure `.env` usage for storing sensitive credentials
- ✅ Local development setup using Docker Compose

---

## 📁 Project Structure
pickup-zone-optimization/
├── .env ← Database credentials (never pushed to Git!)
├── .gitignore
├── requirements.txt
├── README.md
├── streamlit_app/
│ ├── app.py ← Main Streamlit dashboard
│ └── utils/
│ └── db.py ← SQL connection helper
├── data/
├── notebooks/
├── sql/
└── assets/


---

## 📊 Dashboard Outputs

- 🗺️ **Interactive map** showing pickup zone clusters
- 📈 **Time-series plots** for pickup demand trends
- 📌 **Strategic zone detection** based on POI and trip frequency

---

## 🔗 Dataset
- Simulated ride-hailing trip data
- All data is stored in a Supabase PostgreSQL database (`public.jakarta_ride_trips`)

---

## 🚀 Run the App (Locally with Docker Compose)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pickup-zone-optimization.git
   cd pickup-zone-optimization

🧪 Run Without Docker (Dev mode)
pip install -r requirements.txt
streamlit run streamlit_app/app.py

🛡️ Security
✅ .env file is used to securely store database credentials

✅ Database connections are encrypted with SSL (sslmode=require)

✅ .env is listed in .gitignore and never pushed to GitHub

🌐 Deployment
Ready for deployment on:

🔹 Streamlit Cloud

🔹 GitHub Actions (CI/CD)