import streamlit as st
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point, LineString
from pyproj import Transformer
from utils import load_model, impute_salinity, fit_tide_model_for_date, get_tide_level

plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "font.size": 5
})

# 页面设置
st.set_page_config(page_title="Ocean Salinity Imputation System", layout="wide")
st.title("Fort Pierce Inlet Ocean Salinity Imputation System")
# st.markdown("This application uses a pre-trained model to estimate salinity based on NOAA tide data and spatiotemporal coordinates.")


# === Single-point simulation ===
st.header("Single Point Salinity Imputation")

# 用户输入
st.sidebar.header("Input Parameters")
selected_date = st.sidebar.date_input("Select Date", value=datetime(2016, 6, 6).date())
selected_time = st.sidebar.time_input("Select Time", value=time(12, 0))
latitude = st.sidebar.number_input("Latitude (°N)", value=27.4689, format="%.4f")
longitude = st.sidebar.number_input("Longitude (°E)", value=-80.2963, format="%.4f")
datetime_input = datetime.combine(selected_date, selected_time)

# 预测流程触发
if st.button("Run Imputation Model"):
    st.info("⏳ Retrieving NOAA tide data and fitting sinusoidal model...")
    tide_level = get_tide_level(datetime_input)

    if tide_level is None:
        st.error("❌ Failed to retrieve tide data. Please check the date or NOAA API availability.")
    else:
        st.success(f"✅ Tide level successfully estimated: {tide_level:.2f} feet")

        # 加载模型并预测
        st.info("🔧 Loading model and performing inference...")
        model = load_model()
        salinity = impute_salinity(model, datetime_input, latitude, longitude, tide_level)

        st.subheader("Imputation Result")
        st.metric(label="Salinity  (PSU)", value=f"{salinity:.2f}")
        st.caption(f"Imputation on {datetime_input}, Latitude (°N): {latitude:.2f}, Longitude (°E): {longitude:.2f}.")

# === 3-hour simulation ===
# if st.button("Simulate 3-Hour Salinity"):
#     st.info("⏳ Simulating ocean drift and predicting salinity every 10 minutes...")

#     # 加载并保持为经纬度坐标系（EPSG:4326）
#     land = gpd.read_file("ne_10m_land.shp").to_crs(epsg=4326)

#     def generate_ocean_point(lat0, lon0, max_drift=0.05, max_attempts=100):
#         """生成一个不落在陆地上的点，最大漂移距离约为 max_drift（单位：度）"""
#         for _ in range(max_attempts):
#             angle = np.random.uniform(0, 5 * np.pi)
#             radius = np.random.uniform(0, max_drift)
#             dlat = radius * np.sin(angle)
#             dlon = radius * np.cos(angle)
#             new_lat = lat0 + dlat
#             new_lon = lon0 + dlon
#             pt = Point(new_lon, new_lat)
#             if not land.contains(pt).any():
#                 return new_lat, new_lon
#         # 如果尝试多次都失败，返回原始点
#         return lat0, lon0

#     timestamps = [datetime_input + timedelta(minutes=10 * i) for i in range(18)]

#     # Simulate drifting coordinates
#     coords = [(latitude, longitude)]
#     for _ in range(17):
#         prev_lat, prev_lon = coords[-1]
#         new_lat, new_lon = generate_ocean_point(prev_lat, prev_lon)
#         coords.append((new_lat, new_lon))
    
#     # Predict each point
#     model = load_model()
#     results = []
#     for t, (lat, lon) in zip(timestamps, coords):
#         tide = get_tide_level(t)
#         if tide is None:
#             tide = 0.5
#         salinity = impute_salinity(model, t, lat, lon, tide)
#         results.append({
#             "Time": t.strftime("%H:%M"),
#             "Latitude": f"{lat:.5f}",
#             "Longitude": f"{lon:.5f}",
#             "Tide (m)": f"{tide:.2f}",
#             "Salinity (PSU)": f"{salinity:.2f}"
#         })

#     df = pd.DataFrame(results)
#     st.subheader("3-Hour Simulated Salinity Drift")
#     st.dataframe(df)
#     st.line_chart(df.set_index("Time")["Salinity (PSU)"])

#     # Create GeoDataFrame from prediction results
#     geometry = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]
#     gdf_result = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")

#     # Project to Web Mercator for basemap
#     gdf_result = gdf_result.to_crs(epsg=3857)
#     # 生成 LineString 连接所有点（注意：先按顺序提取经纬度）
#     path_line = LineString([Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])])
#     gdf_path = gpd.GeoDataFrame(geometry=[path_line], crs="EPSG:4326").to_crs(epsg=3857)

#     # Plot with basemap
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # 绘制点
#     gdf_result.plot(
#         column="Salinity (PSU)",
#         ax=ax,
#         cmap="viridis",
#         markersize=20,
#         alpha=0.9,
#         legend=False 
#     )

#     # 绘制路径线（黑色透明）
#     gdf_path.plot(ax=ax, color="black", linewidth=2, alpha=0.6)

#     # 手动添加 colorbar
#     norm = plt.Normalize(vmin=gdf_result.iloc[:, -2].min(), vmax=gdf_result.iloc[:, -2].max())
#     sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
#     sm._A = []  

#     from pyproj import Transformer
#     transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
#     x_min, y_min = transformer.transform(-81.0, 26.5)
#     x_max, y_max = transformer.transform(-79.5, 28.0)
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)

#     # 添加底图和图形美化
#     cx.add_basemap(ax, source=cx.providers.Esri.WorldStreetMap, zoom=8)
#     ax.grid(True, linestyle="--", color="grey", alpha=0.5)
#     ax.set_title("Predicted Salinity Drift Path (3-Hour Simulation)")

#     # 在 Streamlit 中显示
#     st.subheader("Salinity Distribution Map with Drift Path")
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.pyplot(fig, bbox_inches='tight')


# === Batch upload imputation ===
st.header("Batch Imputation from Uploaded CSV")
uploaded_file = st.file_uploader("Upload a CSV file with columns: `time`, `latitude`, `longitude`", type="csv")

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)[["time", "latitude", "longitude"]].copy()
        df_input["time"] = pd.to_datetime(df_input["time"], errors="coerce")
        df_input = df_input.dropna(subset=["time", "latitude", "longitude"])
        unique_dates = df_input["time"].dt.date.unique()
        if len(unique_dates) != 1:
            st.error("Uploaded CSV must contain timestamps from only one date.")
            st.stop()

        date_for_tide = unique_dates[0]
        tide_func = fit_tide_model_for_date(date_for_tide)
        st.success(f"Tide model fitted for {date_for_tide}")

        model = load_model()
        tide_levels = []
        salinities = []

        for _, row in df_input.iterrows():
            dt = row["time"]
            lat = row["latitude"]
            lon = row["longitude"]
            t_hour = dt.hour + dt.minute / 60
            try:
                tide = tide_func(t_hour)
            except:
                tide = 0.5
            tide_levels.append(tide)
            sal = impute_salinity(model, dt, lat, lon, tide)
            salinities.append(sal)

        df_input["tide_level"] = tide_levels
        df_input["predicted_salinity"] = salinities

        st.subheader("Imputation Results")
        

        gdf_result = gpd.GeoDataFrame(df_input.copy(), geometry=[Point(lon, lat) for lon, lat in zip(df_input["longitude"], df_input["latitude"])]).set_crs("EPSG:4326").to_crs(epsg=3857)
        path_line = LineString(gdf_result.geometry.tolist())
        gdf_path = gpd.GeoDataFrame(geometry=[path_line], crs="EPSG:3857")

        fig, ax = plt.subplots(figsize=(5, 5))
        gdf_result.plot(column="predicted_salinity", ax=ax, cmap="viridis", markersize=10, alpha=0.9, legend=False)
        # 手动添加 colorbar
        norm = plt.Normalize(vmin=gdf_result.iloc[:, -2].min(), vmax=gdf_result.iloc[:, -2].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm._A = []  

        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
        cbar.set_label("Salinity (PSU)")
        # gdf_path.plot(ax=ax, color="black", linewidth=2, alpha=0.6)
        x_min, y_min = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(-80.35, 27.43)
        x_max, y_max = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(-80.25, 27.505)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldStreetMap, zoom=9)
        ax.set_title("Imputated Salinity")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig, bbox_inches='tight')


        csv_out = df_input.to_csv(index=False).encode("utf-8")
        st.download_button("Download Imputated CSV", csv_out, "salinity_predictions.csv", "text/csv")

        st.dataframe(df_input)

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
