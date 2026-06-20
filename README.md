👋 Hi there! I'm Mahardi Setyoso Pratomo

---

## 🎯 Geospatial Data Engineer | Adding the WHERE to Data

Geospatial Data Engineer (ex-Grab, 8 yrs map ops) building production-grade
spatial data pipelines on GCP — Terraform, Kestra, dbt, PostGIS, H3. Most
data answers what, who, when, and why. I work on the dimension most teams
skip: **where**.

---

## 🚀 About Me

- 🎓 **Education:** B.Sc. in Geographic Information System, Universitas Gadjah Mada
- 📍 **Location:** Jakarta, Indonesia
- 🧭 **Background:** 8 years operating location data at Grab scale — POI
  collection, routing & pricing data, IoT deployment across 200+ sites,
  venue mapping that cut ride cancellations 70–84%
- 🎯 **Target Position:** Geospatial Data Engineer / Analytics Engineer

---

## 🛠️ Technical Skills

**Cloud & Infrastructure:**
GCP (BigQuery, GCS, Dataproc) • Terraform • Docker

**Orchestration & Transformation:**
Kestra • dbt • PySpark

**Geospatial:**
H3 • PostGIS • GeoPandas • Shapely • QGIS

**Languages:**
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) ![SQL](https://img.shields.io/badge/-SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

**Business & Leadership:**  
📈 Project Management • 🤝 Stakeholder Management • 🧠 Strategic Thinking

---

💼 Featured Project

### NYC Taxi Supply–Demand Gap Analysis
*A production-grade geospatial pipeline analyzing a full year of NYC taxi
data to find where demand outruns supply.*

- **Business Question:** Where does taxi demand go unmet across NYC, and
  what's the cost of that gap?
- **Pipeline:** Terraform-provisioned GCP infra → Kestra-orchestrated
  ingestion (12 months TLC trip data, 35M+ rows) → Dataproc/PySpark H3
  enrichment → BigQuery + dbt transformation → PostGIS spatial joins →
  Streamlit dashboard
- **Key Finding:** 13 chronically undersupplied zones vs. 159 oversupplied
  — an estimated $81.6M/year in unmet demand. JFK and LaGuardia lead, with
  fleet misallocation (not city-wide shortage) as the root cause.
- **Engineering rigor:** 50+ architectural decisions documented with
  trade-offs, including a caught-and-fixed aggregation bug (averaging
  ratios vs. rate-of-totals) with regression tests added in dbt.
- **Tech Stack:** GCP • Terraform • Kestra • PySpark • dbt • PostGIS • H3 • Streamlit
- 🔗 [Live Dashboard](https://nyctaxisupplydemandanalysis.streamlit.app)
- 🔗 [Repository & Decision Log](https://github.com/mahardisetyoso/nyc_taxi_supply_demand_analysis)

### OSM Road Density Analysis
*Spatial correlation analysis testing whether road infrastructure explains
the supply-demand gap above.*

- **Business Question:** Are undersupplied zones undersupplied because
  they're hard to reach?
- **Method:** Loaded 72,005 OSM drivable road segments to PostGIS,
  computed road segment density per zone via `ST_Intersects`
- **Key Finding:** Undersupplied zones average *more* road segments than
  oversupplied zones (113.8 vs. 95.1) — road access is not the bottleneck.
  This strengthens the fleet misallocation argument: taxis are avoiding
  the most accessible, highest-demand zones.
- **Tech Stack:** PostGIS • OSM (Geofabrik) • SQL
- 🔗 Part of the NYC Taxi repository above

---

🔭 Currently Building

Continuing the NYC Taxi Supply-Demand project: bivariate choropleth (zone
status × road density), time-of-day demand heatmaps, and an expanded
findings narrative on the dashboard.

---

📚 Continuous Learning

- 🔬 **Algorit.ma Alumni** – Libra Data Science Workshop with Python and R  
- ⚡ **Algorit.ma Alumni** – Developing Data ETL Application with PySpark and Kafka

---

📂 Other / Earlier Projects

*Earlier Python and data analysis projects — kept for reference, not
representative of current focus.*

- **Geospatial Customer Segmentation for Mobility** — clustering customers
  by location and travel behavior (Python, Scikit-learn, GeoPandas)
- **Pickup Zone Optimization for Ride-Hailing** — identifying optimal
  pickup zones to reduce driver-passenger matching time (Python, Folium, QGIS)
- **Predictive Analytics for Demand Forecasting** — ride-hailing demand
  prediction using spatial and temporal features (Scikit-learn, XGBoost, H3-Py)
- **GDP–Education Correlation Visualization** — interactive map correlating
  provincial GDP with education distribution in Indonesia (GeoPandas, Streamlit)
- **Geohash Extractor** — one-click geohash extraction tool (Python, Streamlit)

---

📊 GitHub Stats

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=mahardisetyoso&show_icons=true&theme=radical)

---

## 💡 Open to Opportunities

- 🔍 **Actively seeking:** Geospatial Data Engineering
- 🚀 **Specialization:** Data Engineering • Tech Innovation • Location Intelligence
- 🌟 **Ready to:** Transform geospatial data into actionable business insights

---

---

> *"Bridging the gap between geospatial technology and business success, one data point at a time."* 🌍✨

---

⭐ **If you find my work interesting, please consider giving my repositories a star!**

🤝 Let's Connect!

- 💼 [LinkedIn](https://linkedin.com/in/mahardisetyoso)
- 📧 mahardisetyoso@gmail.com