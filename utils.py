import torch
import numpy as np
import requests
import joblib
from datetime import datetime
from scipy.optimize import curve_fit
import torch.nn as nn
import os
# --------------------------------------------
# 模型定义
# --------------------------------------------
class MLPImputer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MLPImputer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

# --------------------------------------------
# 加载 scaler 和模型
# --------------------------------------------
x_scaler = joblib.load(os.path.join("model", "x_scaler.pkl"))
y_scaler = joblib.load(os.path.join("model", "y_scaler.pkl"))

def load_model(path=os.path.join("model", "mlp_model.pt")):
    model = MLPImputer(input_dim=11, hidden_dim=64)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# --------------------------------------------
# NOAA 潮汐数据获取 + 正弦拟合预测指定时刻潮位
# --------------------------------------------
def get_tide_level(dt, station_id="8722212"):
    """
    Retrieving NOAA tide data and fitting sinusoidal model
    """
    date_str = dt.strftime("%Y%m%d")
    url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "product": "predictions",
        # "application": "salinity_imputer",
        "begin_date": date_str,
        "end_date": date_str,
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "gmt",
        "interval": "hilo",
        "units": "english",
        "format": "json"
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data_json = res.json()
    
        if "predictions" not in data_json:
            print("NOAA API return error：", data_json)
            return None
        data = res.json()["predictions"]

        times = [datetime.strptime(d["t"], "%Y-%m-%d %H:%M") for d in data]
        values = [float(d["v"]) for d in data]
        hours = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
        values = np.array(values)

        # 正弦拟合函数
        def tide_func(t, A, T, phi, c):
            return A * np.sin(2 * np.pi / T * t + phi) + c

        guess = [0.5, 12.0, 0.0, np.mean(values)]
        popt, _ = curve_fit(tide_func, hours, values, p0=guess)

        # 预测目标时间的潮位
        hour_of_day = (dt - datetime(dt.year, dt.month, dt.day)).total_seconds() / 3600
        predicted = tide_func(hour_of_day, *popt)
        return float(predicted)

    except Exception as e:
        print("Tide data fetch or fit failed:", e)
        return None

# --------------------------------------------
# 构造输入并调用模型进行盐度插值
# --------------------------------------------
def impute_salinity(model, dt: datetime, lat: float, lon: float, tide_level: float):
    hour = dt.hour
    minute = dt.minute
    day = dt.day
    month = dt.month

    features = [
        lat, lon,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * minute / 60),
        np.cos(2 * np.pi * minute / 60),
        np.sin(2 * np.pi * day / 31),
        np.cos(2 * np.pi * day / 31),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        tide_level
    ]

    x = np.array([features])
    x_scaled = x_scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_scaled = model(x_tensor).numpy()
    y = y_scaler.inverse_transform(y_scaled)
    return y.item()

def fit_tide_model_for_date(date, station_id="8722212"):
    date_str = date.strftime("%Y%m%d")
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "product": "predictions",
        # "application": "salinity_imputer",
        "begin_date": date_str,
        "end_date": date_str,
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "gmt",
        "interval": "hilo",
        "units": "english",
        "format": "json"
    }
    res = requests.get(url, params=params)
    data = res.json()["predictions"]
    times = [datetime.strptime(d["t"], "%Y-%m-%d %H:%M") for d in data]
    values = [float(d["v"]) for d in data]
    hours = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
    values = np.array(values)

    def tide_func(t, A, phi, c):
        T = 12.42
        return A * np.sin(2 * np.pi / T * t + phi) + c

    popt, _ = curve_fit(tide_func, hours, values, p0=[0.5, 0.0, np.mean(values)])
    return lambda t: tide_func(t, *popt)
