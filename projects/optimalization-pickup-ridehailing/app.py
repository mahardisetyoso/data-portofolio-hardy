import sys, platform, subprocess

def _v(pkg):
    try:
        import importlib
        m = importlib.import_module(pkg)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"ERROR: {e}"

print("PY:", sys.version)
print("Platform:", platform.platform())
for pkg in ["geopandas","pandas","shapely","pyproj","rtree","pyogrio"]:
    print(pkg, _v(pkg))
# ============================
# STREAMLIT: Live from Supabase (no files)
# ============================
import os
import datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

from shapely import wkb
from sqlalchemy import create_engine, text
from streamlit_folium import st_folium
import folium
from folium.features import GeoJsonTooltip
from branca.colormap import LinearColormap

# Optional modeling
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import pulp

st.set_page_config(page_title="Ride-Hailing Pickup Optimization — Live", layout="wide")
st.title("🚖 Ride-Hailing Pickup Zone Optimization — Live (Supabase)")

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========================
# 0) CONFIG & CACHING
# =========================
@st.cache_resource
def get_engine():
    url = st.secrets.get("SUPABASE_DB_URL") or os.environ.get("SUPABASE_DB_URL")
    if not url:
        st.error("Environment variable or Streamlit secret `SUPABASE_DB_URL` not set.")
        st.stop()
    return create_engine(url)

@st.cache_data(show_spinner=False, ttl=60*5)
def fetch_df(sql: str, params=None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params or {})
    # Sanitize: memoryview -> bytes (picklable for cache)
    for c in df.columns:
        if df[c].dtype == "object" and df[c].map(lambda x: isinstance(x, memoryview)).any():
            df[c] = df[c].map(lambda x: bytes(x) if isinstance(x, memoryview) else x)
    return df

def wkb_to_geom(series: pd.Series) -> pd.Series:
    def _load(x):
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            return wkb.loads(x)
        if isinstance(x, memoryview):
            return wkb.loads(bytes(x))
        if isinstance(x, str):  # hex text
            return wkb.loads(bytes.fromhex(x), hex=True)
        return None
    return series.apply(_load)

# =========================
# 1) SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.header("Filters")
    start_date = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=7))
    end_date   = st.date_input("End",   value=dt.date.today())

    hours = st.multiselect("Hours (pickup)", options=list(range(24)), default=[7,8,9])
    weekday_only = st.checkbox("Weekday only", value=False)
    weekend_only = st.checkbox("Weekend only", value=False)
    layer_kind   = st.selectbox("Demand layer", ["pickup", "dropoff", "both"], index=2)
    normalize    = st.checkbox("Normalize per km²", value=True)

    st.markdown("---")
    st.subheader("Modeling")
    do_kmeans = st.checkbox("Show KMeans candidates (peak hours)", value=True)
    do_mclp   = st.checkbox("Run MCLP (experimental)", value=True)
    L = st.slider("Facilities (L)", 5, 60, 15, step=1)
    R = st.slider("Radius (meters)", 100, 1500, 500, step=50)

# =========================
# 2) MASTER LAYERS (Admin4 + Geohash7)
# =========================
admin_sql = """
SELECT "NAME_1","NAME_2","NAME_3","NAME_4",
       encode(ST_AsBinary(geometry), 'hex') AS geom_hex
FROM id_adm_4
WHERE "NAME_1" = 'Jakarta Raya';
"""
df_admin = fetch_df(admin_sql)
df_admin["geometry"] = wkb_to_geom(df_admin["geom_hex"])
df_admin.drop(columns=["geom_hex"], inplace=True)
gdf_admin = gpd.GeoDataFrame(df_admin, geometry="geometry", crs="EPSG:4326")

gh_sql = """
SELECT geohash_list,
       encode(ST_AsBinary(geometry), 'hex') AS geom_hex
FROM id_jkt_geohash7;
"""
df_gh = fetch_df(gh_sql)
df_gh["geometry"] = wkb_to_geom(df_gh["geom_hex"])
df_gh.drop(columns=["geom_hex"], inplace=True)
gdf_geohash = gpd.GeoDataFrame(df_gh, geometry="geometry", crs="EPSG:4326")

# =========================
# 3) DEMAND AGGREGATION (direct SQL)
# =========================
clauses = ["pickup_datetime >= :start_dt", "pickup_datetime <= :end_dt"]
params = {"start_dt": dt.datetime.combine(start_date, dt.time.min),
          "end_dt":   dt.datetime.combine(end_date,   dt.time.max)}

if hours:
    clauses.append("EXTRACT(HOUR FROM pickup_datetime AT TIME ZONE 'Asia/Jakarta') = ANY(:hours)")
    params["hours"] = hours

if weekday_only and not weekend_only:
    clauses.append("EXTRACT(ISODOW FROM pickup_datetime AT TIME ZONE 'Asia/Jakarta') BETWEEN 1 AND 5")
elif weekend_only and not weekday_only:
    clauses.append("EXTRACT(ISODOW FROM pickup_datetime AT TIME ZONE 'Asia/Jakarta') IN (6,7)")

where_sql = " AND ".join(clauses)

if layer_kind == "pickup":
    agg_sql = f"""
    SELECT p.geohash_list AS geohash_list, COUNT(*)::int AS pickup_count, 0::int AS dropoff_count
    FROM jakarta_ride_trips t
    LEFT JOIN id_jkt_geohash7 p
      ON ST_Intersects(p.geometry, ST_SetSRID(ST_Point(t.pickup_longitude, t.pickup_latitude),4326))
    WHERE {where_sql}
    GROUP BY p.geohash_list;
    """
elif layer_kind == "dropoff":
    agg_sql = f"""
    SELECT d.geohash_list AS geohash_list, 0::int AS pickup_count, COUNT(*)::int AS dropoff_count
    FROM jakarta_ride_trips t
    LEFT JOIN id_jkt_geohash7 d
      ON ST_Intersects(d.geometry, ST_SetSRID(ST_Point(t.dropoff_longitude, t.dropoff_latitude),4326))
    WHERE {where_sql}
    GROUP BY d.geohash_list;
    """
else:  # both
    agg_sql = f"""
    WITH p AS (
      SELECT p.geohash_list, COUNT(*)::int AS pickup_count
      FROM jakarta_ride_trips t
      LEFT JOIN id_jkt_geohash7 p
        ON ST_Intersects(p.geometry, ST_SetSRID(ST_Point(t.pickup_longitude, t.pickup_latitude),4326))
      WHERE {where_sql}
      GROUP BY p.geohash_list
    ),
    d AS (
      SELECT d.geohash_list, COUNT(*)::int AS dropoff_count
      FROM jakarta_ride_trips t
      LEFT JOIN id_jkt_geohash7 d
        ON ST_Intersects(d.geometry, ST_SetSRID(ST_Point(t.dropoff_longitude, t.dropoff_latitude),4326))
      WHERE {where_sql}
      GROUP BY d.geohash_list
    )
    SELECT COALESCE(p.geohash_list, d.geohash_list) AS geohash_list,
           COALESCE(p.pickup_count,0)::int AS pickup_count,
           COALESCE(d.dropoff_count,0)::int AS dropoff_count
    FROM p FULL OUTER JOIN d ON p.geohash_list = d.geohash_list;
    """

agg = fetch_df(agg_sql, params=params)
if agg.empty:
    st.warning("No data for the selected filters.")
    st.stop()

agg["total_count"] = agg["pickup_count"] + agg["dropoff_count"]
gdf = gdf_geohash.merge(agg, on="geohash_list", how="left").fillna(0)

if normalize:
    g_m = gdf.to_crs(3857)
    gdf["area_km2"] = (g_m.geometry.area / 1_000_000.0).replace({0: np.nan})
    for base in ["pickup_count","dropoff_count","total_count"]:
        gdf[base + "_per_km2"] = (gdf[base] / gdf["area_km2"]).replace([np.inf,-np.inf], np.nan).fillna(0)

value_col = {
    ("pickup", False): "pickup_count",
    ("dropoff", False): "dropoff_count",
    ("both", False): "total_count",
    ("pickup", True): "pickup_count_per_km2",
    ("dropoff", True): "dropoff_count_per_km2",
    ("both", True): "total_count_per_km2",
}[(layer_kind, normalize)]

# =========================
# 4) MAPS (demand + outline + optional layers)
# =========================
tab_map, tab_tables, tab_model, tab_eval = st.tabs(["🗺️ Map", "📋 Tables", "🧩 Modeling", "📏 Evaluation"])

# replace your existing helper with this version
def make_choropleth(gh_gdf, value_col, title, add_outline=True):
    """
    Draw a Folium choropleth using gh_gdf[value_col].
    NOTE: parameter name is `value_col` (not `val_col`).
    """
    v = gh_gdf[value_col].astype(float)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    cmap = LinearColormap(["green","yellow","red"], vmin=vmin, vmax=vmax)
    cmap.caption = f"{title}"

    m = folium.Map(location=[-6.2,106.8], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(
        data=gh_gdf.to_json(),
        name=title,
        style_function=lambda f: {
            "fillColor": cmap(f["properties"].get(value_col, 0.0)),
            "color": "#555",
            "weight": 0.4,
            "fillOpacity": 0.75
        },
        tooltip=GeoJsonTooltip(
            fields=[c for c in ["geohash_list","pickup_count","dropoff_count","total_count", value_col]
                    if c in gh_gdf.columns],
            aliases=["Geohash","Pickup","Dropoff","Total","Value"],
            sticky=True, localize=True
        )
    ).add_to(m)

    if add_outline:
        # assumes gdf_admin exists in outer scope
        folium.GeoJson(
            data=gdf_admin.to_json(),
            name="Admin4 outline",
            style_function=lambda f: {"color":"#222","weight":0.6,"fillOpacity":0}
        ).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

with tab_map:
    st.subheader("Demand Choropleth")
    the_map = make_choropleth(
        gh_gdf=gdf,
        value_col=value_col,
        title=f"Demand ({value_col})"
        )
    st_map = st_folium(the_map, height=600, use_container_width=True)

# =========================
# 5) TABLES & DISTRIBUTIONS
# =========================
with tab_tables:
    st.subheader("Top Geohash cells")
    show_cols = [c for c in ["geohash_list","pickup_count","dropoff_count","total_count",
                             "pickup_count_per_km2","dropoff_count_per_km2","total_count_per_km2"] if c in gdf.columns]
    st.dataframe(
        gdf.drop(columns="geometry").sort_values(value_col, ascending=False)[show_cols].head(50),
        use_container_width=True
    )

    st.subheader("Histogram")
    hist_data, _ = np.histogram(gdf[value_col].astype(float), bins=30)
    st.bar_chart(pd.DataFrame({"bin": np.arange(len(hist_data)), "count": hist_data}).set_index("bin"))

# =========================
# 6) MODELING (KMeans candidates + MCLP)
# =========================
def kmeans_candidates(hours_like, tppc=25, min_clusters=8, max_clusters=120, min_cluster_size_floor=3):
    clauses2 = ["pickup_datetime >= :start_dt", "pickup_datetime <= :end_dt",
                "pickup_latitude IS NOT NULL", "pickup_longitude IS NOT NULL"]
    par2 = {"start_dt": params["start_dt"], "end_dt": params["end_dt"]}

    if hours_like:
        clauses2.append("EXTRACT(HOUR FROM pickup_datetime AT TIME ZONE 'Asia/Jakarta') = ANY(:hh)")
        par2["hh"] = list(hours_like)
    if weekday_only and not weekend_only:
        clauses2.append("EXTRACT(ISODOW FROM pickup_datetime AT TIME ZONE 'Asia/Jakarta') BETWEEN 1 AND 5")
    elif weekend_only and not weekday_only:
        clauses2.append("EXTRACT(ISODOW FROM pickup_datetime AT TIME ZONE 'Asia/Jakarta') IN (6,7)")

    q = f"SELECT pickup_longitude AS lon, pickup_latitude AS lat FROM jakarta_ride_trips WHERE {' AND '.join(clauses2)};"
    pts = fetch_df(q, par2)
    if pts.empty or len(pts) < 2:
        return gpd.GeoDataFrame(columns=["geometry","cluster_size","candidate_type"], geometry="geometry", crs="EPSG:4326")

    n = len(pts)
    est_clusters = max(min_clusters, min(max_clusters, max(1, int(n / max(5, tppc)))))
    est_clusters = min(est_clusters, n)

    X = pts[["lon","lat"]].to_numpy()
    km = KMeans(n_clusters=est_clusters, random_state=42, n_init="auto").fit(X)
    pts["cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=["lon","lat"])
    sizes = pts.groupby("cluster").size().rename("cluster_size").reset_index()
    centers = centers.join(sizes.set_index("cluster"), how="left")

    thr = max(min_cluster_size_floor, int(np.percentile(centers["cluster_size"], 10)))
    kept = centers[centers["cluster_size"] >= thr].copy()
    if kept.empty: kept = centers.copy()

    out = gpd.GeoDataFrame(kept, geometry=gpd.points_from_xy(kept["lon"], kept["lat"]), crs="EPSG:4326")
    out["candidate_type"] = f"kmeans_peak_{min(hours) if hours else '-'}-{max(hours) if hours else '-'}"
    return out[["geometry","cluster_size","candidate_type"]]

def to_centroids(gh_gdf, metric_col):
    h = gh_gdf[gh_gdf[metric_col] > 0].copy()
    h = h.to_crs(4326)
    h["geometry"] = h.geometry.centroid
    return gpd.GeoDataFrame(h[["geometry", metric_col]].rename(columns={metric_col: "weight"}), geometry="geometry", crs="EPSG:4326")

def solve_mclp(demand_points: gpd.GeoDataFrame, candidates: gpd.GeoDataFrame, L=10, R=500, weight_col="weight"):
    if demand_points.empty or candidates.empty:
        return [], 0.0, 0.0, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    dm = demand_points.to_crs(3857).reset_index(drop=True)
    cm = candidates.to_crs(3857).reset_index(drop=True)
    dem_xy = np.c_[dm.geometry.x, dm.geometry.y]
    fac_xy = np.c_[cm.geometry.x, cm.geometry.y]
    tree = BallTree(fac_xy, metric="euclidean")
    neighbors = tree.query_radius(dem_xy, r=R)

    prob = pulp.LpProblem("MCLP", sense=pulp.LpMaximize)
    J = range(len(cm)); I = range(len(dm))
    y = pulp.LpVariable.dicts("y", J, 0, 1, cat="Binary")
    z = pulp.LpVariable.dicts("z", I, 0, 1, cat="Binary")
    w = dm[weight_col].values.astype(float)
    prob += pulp.lpSum(w[i]*z[i] for i in I)
    prob += pulp.lpSum(y[j] for j in J) == L
    for i, Nj in enumerate(neighbors):
        if len(Nj) == 0:
            prob += z[i] == 0
        else:
            prob += z[i] <= pulp.lpSum(y[int(j)] for j in Nj)
    _ = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    sel_idx = [j for j in J if pulp.value(y[j]) > 0.5]
    covered = float(sum(w[i] for i in I if pulp.value(z[i]) > 0.5))
    total = float(w.sum())
    sel_gdf = cm.iloc[sel_idx].copy().to_crs(4326); sel_gdf["selected"] = 1
    return sel_idx, covered, total, sel_gdf

def baseline_topL(demand_points: gpd.GeoDataFrame, L=10, R=500):
    top = demand_points.sort_values("weight", ascending=False).head(L).copy()
    dm = demand_points.to_crs(3857).reset_index(drop=True)
    cm = top.to_crs(3857).reset_index(drop=True)
    dem_xy = np.c_[dm.geometry.x, dm.geometry.y]
    sel_xy = np.c_[cm.geometry.x, cm.geometry.y]
    tree = BallTree(sel_xy, metric="euclidean")
    near = tree.query_radius(dem_xy, r=R)
    covered_mask = np.array([len(n) > 0 for n in near])
    covered = float(dm.loc[covered_mask, "weight"].sum())
    total = float(dm["weight"].sum())
    return top.to_crs(4326), covered, total

def mean_nearest_distance(demand_points: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame):
    if demand_points.empty or facilities.empty:
        return np.nan
    dm = demand_points.to_crs(3857).reset_index(drop=True)
    fm = facilities.to_crs(3857).reset_index(drop=True)
    dem_xy = np.c_[dm.geometry.x, dm.geometry.y]
    fac_xy = np.c_[fm.geometry.x, fm.geometry.y]
    tree = BallTree(fac_xy, metric="euclidean")
    dists, _ = tree.query(dem_xy, k=1)
    return float(np.mean(dists))

with tab_model:
    st.subheader("Candidates & Optimization")
    demand_points = to_centroids(gdf, value_col)
    st.caption(f"Demand points: {len(demand_points):,} | total weight: {demand_points['weight'].sum():,.0f}")

    cand_geo = demand_points[["geometry"]].copy()
    cand_geo["candidate_type"] = "geohash_centroid"

    if do_kmeans:
        cand_km = kmeans_candidates(hours_like=hours, tppc=25)
        st.write(f"KMeans candidates: {len(cand_km)}")
    else:
        cand_km = gpd.GeoDataFrame(columns=["geometry","candidate_type","cluster_size"], geometry="geometry", crs="EPSG:4326")

    choice = st.radio("Candidate set", ["Geohash", "KMeans", "Hybrid"], horizontal=True)
    if choice == "Geohash":
        candidates = cand_geo
    elif choice == "KMeans":
        candidates = cand_km if len(cand_km) > 0 else cand_geo
    else:
        candidates = pd.concat([cand_geo, cand_km]) if len(cand_km) > 0 else cand_geo
        candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs="EPSG:4326")

    baseline_pts, cov_b, tot_b = baseline_topL(demand_points, L=L, R=R)
    _, cov_m, tot_m, sol_pts   = solve_mclp(demand_points, candidates, L=L, R=R)

    cov_b_pct = cov_b / tot_b if tot_b > 0 else 0.0
    cov_m_pct = cov_m / tot_m if tot_m > 0 else 0.0

    st.markdown(f"**Coverage (Baseline vs MCLP)**: {cov_b_pct: .1%} → **{cov_m_pct: .1%}** (Δ {(cov_m_pct-cov_b_pct)*100: .1f} pp)")

    d_baseline = mean_nearest_distance(demand_points, baseline_pts)
    d_mclp     = mean_nearest_distance(demand_points, sol_pts)
    st.markdown(f"**Mean nearest distance (proxy wait-time)**: {d_baseline:,.0f} m → **{d_mclp:,.0f} m** (Δ {d_baseline-d_mclp:,.0f} m)")

    map2 = make_choropleth(
        gh_gdf=gdf.assign(weight=gdf[value_col]),
        value_col="weight",
        title=f"Demand ({value_col})"
        )
    for _, r in baseline_pts.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=5, color="#f39c12",
                            fill=True, fill_opacity=0.9, tooltip="Baseline").add_to(map2)
    for _, r in sol_pts.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=6, color="#27ae60",
                            fill=True, fill_opacity=1.0, tooltip="MCLP").add_to(map2)
        buf = gpd.GeoSeries([r.geometry], crs=4326).to_crs(3857).buffer(R).to_crs(4326)
        folium.GeoJson(buf.__geo_interface__, style_function=lambda f: {"color":"#27ae60","weight":1,"fillOpacity":0.05}).add_to(map2)
    st_folium(map2, height=600, use_container_width=True)

# =========================
# 7) EVALUATION (Coverage & Admin comparison)
# =========================
with tab_eval:
    st.subheader("Evaluation")

    df_eval = pd.DataFrame({
        "metric": ["Coverage % (Baseline)", "Coverage % (MCLP)", "Δ Coverage (pp)",
                   "Mean nearest dist (Baseline, m)", "Mean nearest dist (MCLP, m)", "Δ Distance (m)"],
        "value":  [cov_b_pct*100, cov_m_pct*100, (cov_m_pct-cov_b_pct)*100,
                   d_baseline, d_mclp, d_baseline-d_mclp]
    })
    st.dataframe(df_eval, use_container_width=True)

    dm = to_centroids(gdf, value_col)
    dm_3857 = dm.to_crs(3857).reset_index(drop=True)
    if not sol_pts.empty:
        sol_3857 = sol_pts.to_crs(3857).reset_index(drop=True)
        dem_xy = np.c_[dm_3857.geometry.x, dm_3857.geometry.y]
        fac_xy = np.c_[sol_3857.geometry.x, sol_3857.geometry.y]
        tree = BallTree(fac_xy, metric="euclidean")
        near = tree.query_radius(dem_xy, r=R)
        covered_mask = np.array([len(n) > 0 for n in near])
    else:
        covered_mask = np.zeros(len(dm_3857), dtype=bool)

    dm_cov = dm.copy()
    dm_cov["covered"] = covered_mask

    # coverage by Admin4 (weight covered / total weight)
    dm_join = gpd.sjoin(dm_cov.set_crs(4326), gdf_admin[["NAME_4","geometry"]], how="left", predicate="within")
    tmp = dm_join.copy()
    tmp["covered_weight"] = tmp["weight"] * tmp["covered"].astype(int)
    admin_cov = (tmp.groupby("NAME_4")
                    .agg(total_weight=("weight","sum"),
                         covered_weight=("covered_weight","sum"))
                    .reset_index())
    admin_cov["coverage_pct"] = (admin_cov["covered_weight"] / admin_cov["total_weight"]).replace([np.inf,-np.inf], np.nan)

    st.markdown("**Coverage by Kelurahan (Admin4)** — MCLP solution")
    st.dataframe(admin_cov.sort_values("coverage_pct", ascending=False), use_container_width=True)

    vmin, vmax = 0.0, 1.0
    cmap = LinearColormap(["#2ecc71","#f1c40f","#e74c3c"], vmin=vmin, vmax=vmax); cmap.caption = "Admin4 Coverage (MCLP)"

    pr = gdf_admin.merge(admin_cov, on="NAME_4", how="left").fillna({"coverage_pct":0})
    m3 = folium.Map(location=[-6.2,106.8], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(
        pr.to_json(),
        name="Admin4 Coverage",
        style_function=lambda f: {"fillColor": cmap(f["properties"].get("coverage_pct", 0.0)),
                                  "color":"#444", "weight":0.6, "fillOpacity":0.75},
        tooltip=GeoJsonTooltip(
            fields=["NAME_4","coverage_pct","covered_weight","total_weight"],
            aliases=["Kelurahan","Coverage","Covered Weight","Total Weight"],
            sticky=True, localize=True
        )
    ).add_to(m3)
    cmap.add_to(m3)
    folium.LayerControl(collapsed=False).add_to(m3)
    st_folium(m3, height=600, use_container_width=True)

st.caption("All layers update live from Supabase based on your filters. No files are generated.")
