# 🚕 Optimasi Zona Pickup untuk Ride-Hailing

## 📌 Latar Belakang
Layanan ride-hailing di kota padat seperti Jakarta menghadapi tantangan dalam hal efisiensi penjemputan. Titik-titik penjemputan yang tidak optimal menyebabkan waktu tunggu tinggi, pembatalan order, dan ketidakpuasan pengguna. Hal ini juga berdampak pada produktivitas driver dan efisiensi platform.

## 🎯 Business Objective
Menentukan zona pickup yang optimal berdasarkan data simulasi perjalanan, sehingga dapat:
- Mengurangi waktu tunggu penumpang
- Meningkatkan efisiensi rute dan waktu kerja driver
- Memberikan insight strategis bagi platform ride-hailing

## 📊 Success Metrics
- ⏱️ Rata-rata waktu penjemputan < 8 menit
- 📈 Akurasi prediksi permintaan zona pickup > 85%
- 📉 Penurunan cancelation rate > 10% (jika tersedia)

## 👥 Stakeholders
- **User (penumpang)**: ingin cepat dijemput dari lokasi strategis
- **Driver**: ingin mendapat order pickup efisien dan mudah dijangkau
- **Platform**: ingin memaksimalkan efisiensi dan kepuasan pengguna

## 🗂️ Dataset Overview
- 5000 trip simulasi Jakarta
- Informasi pickup, dropoff, waktu, cuaca, tipe kendaraan, zona
- Data tersedia di folder `/data/`

## 🛠️ Tools
- Python, Streamlit, Folium, Geopandas
- PostgreSQL + PostGIS untuk analisis spasial lanjutan
