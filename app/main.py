from __future__ import annotations

from pathlib import Path
from threading import Lock
import base64
import html
import io
import json
import time
from datetime import datetime
import xml.etree.ElementTree as ET

import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Scientific/GIS libraries are imported only when needed.
# The public home page and /map HTML can therefore start immediately on Render.
folium = None
gpd = None
plt = None
FormatStrFormatter = None
np = None
pd = None
ImageOverlay = None
cKDTree = None
intersects = None
points = None
_data_import_lock = Lock()
_plot_import_lock = Lock()
_data_imports_loaded = False
_plot_imports_loaded = False


def ensure_data_imports() -> None:
    """Load only the libraries required for data, IDW, and geometry work."""
    global gpd, np, pd, cKDTree, intersects, points
    global _data_imports_loaded

    if _data_imports_loaded:
        return

    with _data_import_lock:
        if _data_imports_loaded:
            return

        import geopandas as _gpd
        import numpy as _np
        import pandas as _pd
        from scipy.spatial import cKDTree as _cKDTree
        from shapely import intersects as _intersects, points as _points

        gpd = _gpd
        np = _np
        pd = _pd
        cKDTree = _cKDTree
        intersects = _intersects
        points = _points
        _data_imports_loaded = True


def ensure_plot_imports() -> None:
    """Load Matplotlib only for publication-map export or legacy rendering."""
    global folium, plt, FormatStrFormatter, ImageOverlay
    global _plot_imports_loaded

    ensure_data_imports()

    if _plot_imports_loaded:
        return

    with _plot_import_lock:
        if _plot_imports_loaded:
            return

        import folium as _folium
        import matplotlib as _matplotlib
        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from matplotlib.ticker import FormatStrFormatter as _FormatStrFormatter
        from folium.raster_layers import ImageOverlay as _ImageOverlay

        folium = _folium
        plt = _plt
        FormatStrFormatter = _FormatStrFormatter
        ImageOverlay = _ImageOverlay
        _plot_imports_loaded = True


# Backward-compatible name used by data functions.
def ensure_heavy_imports() -> None:
    ensure_data_imports()


# =========================================================
# 1. FastAPI
# =========================================================

app = FastAPI(
    title="Thailand Weather Interpolation Map",
    description=(
        "Interactive Tmin interpolation map "
        "with OpenStreetMap place search"
    ),
)


# =========================================================
# 2. ตำแหน่งไฟล์
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TMD_API_URL = (
    "https://data.tmd.go.th/api/WeatherToday/V2/"
    "?uid=api&ukey=api12345"
)

WEATHER3HOURS_API_URL = (
    "https://data.tmd.go.th/api/Weather3Hours/V2/"
    "?uid=api&ukey=api12345"
)

TMD_API_USER_AGENT = (
    "ThailandTminMap/1.0 "
    "(contact: sukrungsri.s@gmail.com)"
)

THAILAND_FILE = (
    BASE_DIR
    / "data"
    / "thailand.geojson"
)


# =========================================================
# 3. การตั้งค่า Nominatim
# =========================================================

NOMINATIM_SEARCH_URL = (
    "https://nominatim.openstreetmap.org/search"
)

# โปรดเปลี่ยนอีเมลให้เป็นอีเมลผู้ดูแลระบบจริงก่อนนำระบบขึ้นใช้งาน
NOMINATIM_USER_AGENT = (
    "ThailandTminMap/1.0 "
    "(contact: sukrungsri.s@gmail.com)"
)

# เก็บเวลาที่ค้นหาครั้งล่าสุด เพื่อเว้นอย่างน้อย 1 วินาที
nominatim_lock = Lock()
last_nominatim_request_time = 0.0

# Cache ลดการเรียก API ซ้ำ
search_cache: dict[str, dict] = {}

CACHE_SECONDS = 60 * 60
MAX_CACHE_ITEMS = 200


# =========================================================
# Cache สำหรับหน้าแผนที่ ลดเวลาประมวลผลบน Render
# =========================================================

WEATHER_OVERLAY_CACHE_SECONDS = 30 * 60
weather_overlay_cache_lock = Lock()
weather_overlay_cache: dict[str, object] = {
    "created_at": 0.0,
    "value": None,
}

# Lightweight caches used by the browser-driven map.
# Each layer is generated separately and reused for 30 minutes.
LAYER_CACHE_SECONDS = 30 * 60
layer_cache_lock = Lock()
layer_cache: dict[str, dict] = {}
boundary_cache: dict[str, object] = {
    "created_at": 0.0,
    "geojson": None,
    "bounds": None,
}

LAYER_DEFINITIONS = {
    "tmin": {
        "field": "tmin",
        "name": "WeatherToday - Tmin",
        "short_name": "Tmin",
        "unit": "°C",
        "palette": "turbo",
        "source": "today",
    },
    "tmax": {
        "field": "tmax",
        "name": "WeatherToday - Tmax",
        "short_name": "Tmax",
        "unit": "°C",
        "palette": "turbo",
        "source": "today",
    },
    "temperature": {
        "field": "temperature",
        "name": "WeatherToday - Current Temperature",
        "short_name": "Temperature",
        "unit": "°C",
        "palette": "turbo",
        "source": "today",
    },
    "rainfall": {
        "field": "rainfall",
        "name": "WeatherToday - Daily Rainfall",
        "short_name": "Rainfall",
        "unit": "mm",
        "palette": "blues",
        "source": "today",
    },
    "air_temperature_3h": {
        "field": "air_temperature_3h",
        "name": "Weather3Hours - Air Temperature",
        "short_name": "Air Temperature",
        "unit": "°C",
        "palette": "turbo",
        "source": "3hours",
    },
    "rainfall_3h": {
        "field": "rainfall_3h",
        "name": "Weather3Hours - Rainfall",
        "short_name": "Rainfall",
        "unit": "mm",
        "palette": "blues",
        "source": "3hours",
    },
    "rainfall_24h_3h": {
        "field": "rainfall_24h_3h",
        "name": "Weather3Hours - Rainfall 24 Hour",
        "short_name": "Rainfall 24h",
        "unit": "mm",
        "palette": "blues",
        "source": "3hours",
    },
}


# =========================================================
# 4. ฟังก์ชันช่วยอ่าน XML จาก TMD API
# =========================================================

def get_xml_text(
    parent: ET.Element,
    path: str,
) -> str | None:
    """
    อ่านข้อความจาก XML ตาม path ที่กำหนด
    หากไม่พบข้อมูลหรือเป็นข้อความว่าง ให้คืนค่า None
    """

    element = parent.find(path)

    if element is None or element.text is None:
        return None

    value = element.text.strip()

    return value if value else None


def to_float(
    value: str | None,
) -> float | None:
    """
    แปลงข้อความเป็นเลขทศนิยม
    หากแปลงไม่ได้ ให้คืนค่า None
    """

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_observation_datetime(
    datetime_text: str,
) -> str:
    """แปลงเวลาเป็น 16 Jul 2026 07:00:00"""

    if not datetime_text:
        return "Unknown"

    value = str(datetime_text).strip()

    for pattern in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
    ):
        try:
            parsed_datetime = datetime.strptime(
                value,
                pattern,
            )
            return parsed_datetime.strftime(
                "%d %b %Y %H:%M:%S"
            )
        except ValueError:
            continue

    return value


# =========================================================
# 5. อ่านข้อมูลสถานีจาก TMD API
# =========================================================

def load_station_data() -> pd.DataFrame:
    ensure_heavy_imports()
    """
    ดึงข้อมูลสถานีจาก TMD WeatherToday API (XML):
    Temperature, Tmax, Tmin และ Rainfall
    """

    headers = {
        "User-Agent": TMD_API_USER_AGENT,
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    }

    try:
        with httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.get(TMD_API_URL)

        response.raise_for_status()

    except httpx.TimeoutException as error:
        raise RuntimeError(
            "TMD API ใช้เวลาตอบกลับนานเกินไป"
        ) from error

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "TMD API ตอบกลับด้วยข้อผิดพลาด "
            f"{error.response.status_code}"
        ) from error

    except httpx.HTTPError as error:
        raise RuntimeError(
            "ไม่สามารถเชื่อมต่อ TMD API ได้"
        ) from error

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as error:
        raise RuntimeError(
            "ข้อมูลจาก TMD API ไม่ใช่ XML ที่อ่านได้"
        ) from error

    station_records: list[dict] = []

    for station in root.findall(".//Station"):
        station_name = (
            get_xml_text(station, "StationNameEnglish")
            or get_xml_text(station, "StationNameThai")
            or "Weather Station"
        )

        province = (
            get_xml_text(station, "Province")
            or ""
        )

        longitude = to_float(
            get_xml_text(station, "Longitude")
        )
        latitude = to_float(
            get_xml_text(station, "Latitude")
        )

        temperature = to_float(
            get_xml_text(station, "Observation/Temperature")
        )

        tmax = to_float(
            get_xml_text(station, "Observation/MaxTemperature")
        )
        if tmax is None:
            tmax = to_float(
                get_xml_text(station, "Observation/TemperatureMax")
            )
        if tmax is None:
            tmax = to_float(
                get_xml_text(station, "Observation/MaximumTemperature")
            )

        tmin = to_float(
            get_xml_text(station, "Observation/MinTemperature")
        )
        if tmin is None:
            tmin = to_float(
                get_xml_text(station, "Observation/TemperatureMin")
            )
        if tmin is None:
            tmin = to_float(
                get_xml_text(station, "Observation/MinimumTemperature")
            )

        rainfall = to_float(
            get_xml_text(station, "Observation/Rainfall")
        )

        observation_datetime = get_xml_text(
            station,
            "Observation/DateTime",
        )

        if longitude is None or latitude is None:
            continue

        if not (
            96.0 <= longitude <= 107.0
            and 4.0 <= latitude <= 22.0
        ):
            continue

        for value in (temperature, tmax, tmin):
            if value is not None and not (-10.0 <= value <= 50.0):
                value = None

        if rainfall is not None and rainfall < 0:
            rainfall = None

        station_records.append(
            {
                "station": station_name,
                "province": province,
                "longitude": longitude,
                "latitude": latitude,
                "temperature": temperature,
                "tmax": tmax,
                "tmin": tmin,
                "rainfall": rainfall,
                "observation_datetime": (
                    observation_datetime or ""
                ),
            }
        )

    station_data = pd.DataFrame(station_records)

    if station_data.empty:
        raise ValueError(
            "ไม่พบข้อมูลสถานีจาก TMD API"
        )

    station_data = (
        station_data
        .drop_duplicates(
            subset=["station", "longitude", "latitude"],
            keep="last",
        )
        .sort_values(
            by=["station", "latitude", "longitude"]
        )
        .reset_index(drop=True)
    )

    return station_data


def get_first_float(
    parent: ET.Element,
    paths: tuple[str, ...],
) -> float | None:
    """
    อ่านค่าเลขจาก XML โดยลองหลายชื่อแท็ก
    """

    for path in paths:
        value = to_float(
            get_xml_text(parent, path)
        )

        if value is not None:
            return value

    return None


def get_first_text(
    parent: ET.Element,
    paths: tuple[str, ...],
) -> str | None:
    """
    อ่านข้อความจาก XML โดยลองหลายชื่อแท็ก
    """

    for path in paths:
        value = get_xml_text(
            parent,
            path,
        )

        if value is not None:
            return value

    return None


def load_weather3hours_data() -> pd.DataFrame:
    ensure_heavy_imports()
    """
    ดึงข้อมูล Weather3Hours จาก TMD API:
    Temperature, Rainfall และ Rainfall24Hour
    """

    headers = {
        "User-Agent": TMD_API_USER_AGENT,
        "Accept": (
            "application/xml,"
            "text/xml;q=0.9,"
            "*/*;q=0.8"
        ),
    }

    try:
        with httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.get(
                WEATHER3HOURS_API_URL
            )

        response.raise_for_status()

    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Weather3Hours API "
            "ใช้เวลาตอบกลับนานเกินไป"
        ) from error

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Weather3Hours API ตอบกลับด้วยข้อผิดพลาด "
            f"{error.response.status_code}"
        ) from error

    except httpx.HTTPError as error:
        raise RuntimeError(
            "ไม่สามารถเชื่อมต่อ Weather3Hours API ได้"
        ) from error

    try:
        root = ET.fromstring(
            response.content
        )
    except ET.ParseError as error:
        raise RuntimeError(
            "ข้อมูลจาก Weather3Hours API "
            "ไม่ใช่ XML ที่อ่านได้"
        ) from error

    records: list[dict] = []

    for station in root.findall(
        ".//Station"
    ):
        station_name = (
            get_first_text(
                station,
                (
                    "StationNameEnglish",
                    "StationNameThai",
                    "StationName",
                ),
            )
            or "Weather Station"
        )

        province = (
            get_first_text(
                station,
                (
                    "Province",
                    "ProvinceName",
                ),
            )
            or ""
        )

        longitude = get_first_float(
            station,
            (
                "Longitude",
                "Lon",
            ),
        )

        latitude = get_first_float(
            station,
            (
                "Latitude",
                "Lat",
            ),
        )

        air_temperature_3h = get_first_float(
            station,
            (
                "Observation/AirTemperature",
                "Observation/AirTemp",
                "Observation/Temperature",
                "Observation/Temp",
                "AirTemperature",
                "AirTemp",
                "Temperature",
                "Temp",
            ),
        )

        rainfall_3h = get_first_float(
            station,
            (
                "Observation/Rainfall",
                "Observation/Rainfall3Hour",
                "Observation/Rainfall3Hours",
                "Observation/Rainfall3Hr",
                "Observation/Rain3Hr",
                "Rainfall",
                "Rainfall3Hour",
                "Rainfall3Hours",
                "Rainfall3Hr",
                "Rain3Hr",
            ),
        )

        rainfall_24h = get_first_float(
            station,
            (
                "Observation/Rainfall24Hour",
                "Observation/Rainfall24Hours",
                "Observation/Rainfall24Hr",
                "Observation/Rain24Hr",
                "Observation/AccumulatedRainfall24Hour",
                "Rainfall24Hour",
                "Rainfall24Hours",
                "Rainfall24Hr",
                "Rain24Hr",
                "AccumulatedRainfall24Hour",
            ),
        )

        observation_datetime_3h = (
            get_first_text(
                station,
                (
                    "Observation/DateTime",
                    "Observation/ObservationDateTime",
                    "DateTime",
                    "ObservationDateTime",
                ),
            )
            or ""
        )

        if (
            longitude is None
            or latitude is None
        ):
            continue

        if not (
            96.0 <= longitude <= 107.0
            and 4.0 <= latitude <= 22.0
        ):
            continue

        if (
            air_temperature_3h is not None
            and not (
                -10.0
                <= air_temperature_3h
                <= 50.0
            )
        ):
            air_temperature_3h = None

        if (
            rainfall_3h is not None
            and rainfall_3h < 0
        ):
            rainfall_3h = None

        if (
            rainfall_24h is not None
            and rainfall_24h < 0
        ):
            rainfall_24h = None

        records.append(
            {
                "station": station_name,
                "province": province,
                "longitude": longitude,
                "latitude": latitude,
                "air_temperature_3h": air_temperature_3h,
                "rainfall_3h": rainfall_3h,
                "rainfall_24h_3h": rainfall_24h,
                "observation_datetime_3h": (
                    observation_datetime_3h
                ),
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    if dataframe.empty:
        raise ValueError(
            "ไม่พบข้อมูลสถานีจาก Weather3Hours API"
        )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "station",
                "longitude",
                "latitude",
            ],
            keep="last",
        )
        .sort_values(
            by=[
                "station",
                "latitude",
                "longitude",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# =========================================================
# 6. อ่านขอบเขตประเทศไทย
# =========================================================

def load_thailand_boundary() -> gpd.GeoDataFrame:
    ensure_heavy_imports()
    """
    อ่าน thailand.geojson และแปลง CRS เป็น EPSG:4326
    """

    if not THAILAND_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ขอบเขตประเทศไทย:\n"
            f"{THAILAND_FILE}"
        )

    thailand = gpd.read_file(
        THAILAND_FILE
    )

    if thailand.empty:
        raise ValueError(
            "ไฟล์ thailand.geojson ไม่มีข้อมูล"
        )

    if thailand.crs is None:
        thailand = thailand.set_crs(
            epsg=4326
        )
    else:
        thailand = thailand.to_crs(
            epsg=4326
        )

    thailand["geometry"] = (
        thailand.geometry.buffer(0)
    )

    thailand = thailand[
        thailand.geometry.notna()
        & ~thailand.geometry.is_empty
    ].copy()

    if thailand.empty:
        raise ValueError(
            "ไม่พบ geometry ที่ใช้ได้ใน thailand.geojson"
        )

    return thailand


# =========================================================
# 7. IDW Interpolation
# =========================================================

def idw_interpolation(
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    station_value: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    power: float = 2.0,
    nearest_points: int = 8,
) -> np.ndarray:
    """
    ทำ Inverse Distance Weighting
    """

    station_coordinates = np.column_stack(
        (
            station_lon,
            station_lat,
        )
    )

    grid_coordinates = np.column_stack(
        (
            grid_lon.ravel(),
            grid_lat.ravel(),
        )
    )

    tree = cKDTree(
        station_coordinates
    )

    number_of_neighbors = min(
        nearest_points,
        len(station_coordinates),
    )

    distances, indexes = tree.query(
        grid_coordinates,
        k=number_of_neighbors,
    )

    if number_of_neighbors == 1:
        distances = distances[:, np.newaxis]
        indexes = indexes[:, np.newaxis]

    distances = np.maximum(
        distances,
        1e-12,
    )

    weights = 1.0 / np.power(
        distances,
        power,
    )

    nearby_values = station_value[indexes]

    interpolated_values = np.sum(
        weights * nearby_values,
        axis=1,
    ) / np.sum(
        weights,
        axis=1,
    )

    return interpolated_values.reshape(
        grid_lon.shape
    )


# =========================================================
# 8. Mask ประเทศไทย
# =========================================================

def create_thailand_mask(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    thailand: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    True คืออยู่ในประเทศไทย
    False คืออยู่นอกประเทศไทย
    """

    thailand_geometry = (
        thailand.geometry.union_all()
    )

    grid_points = points(
        grid_lon.ravel(),
        grid_lat.ravel(),
    )

    inside_thailand = intersects(
        grid_points,
        thailand_geometry,
    )

    return inside_thailand.reshape(
        grid_lon.shape
    )


# =========================================================
# 9. สร้างภาพเฉดสี
# =========================================================

def build_weather_overlays():
    ensure_plot_imports()
    """
    สร้าง Layer จาก 2 API

    WeatherToday:
    - Tmin
    - Tmax
    - Current Temperature
    - Daily Rainfall

    Weather3Hours:
    - Air Temperature
    - Rainfall
    - Rainfall 24 Hour
    """

    today_columns = [
        "station",
        "province",
        "longitude",
        "latitude",
        "temperature",
        "tmax",
        "tmin",
        "rainfall",
        "observation_datetime",
    ]

    weather3h_columns = [
        "station",
        "province",
        "longitude",
        "latitude",
        "air_temperature_3h",
        "rainfall_3h",
        "rainfall_24h_3h",
        "observation_datetime_3h",
    ]

    today_error = None
    weather3h_error = None

    try:
        today_data = load_station_data()
    except Exception as error:
        today_error = str(error)
        today_data = pd.DataFrame(
            columns=today_columns
        )

    try:
        weather3h_data = load_weather3hours_data()
    except Exception as error:
        weather3h_error = str(error)
        weather3h_data = pd.DataFrame(
            columns=weather3h_columns
        )

    if today_data.empty and weather3h_data.empty:
        raise RuntimeError(
            "ไม่สามารถดึงข้อมูลจากทั้ง WeatherToday "
            "และ Weather3Hours ได้\n"
            f"WeatherToday: {today_error}\n"
            f"Weather3Hours: {weather3h_error}"
        )

    thailand = load_thailand_boundary()

    (
        minimum_lon,
        minimum_lat,
        maximum_lon,
        maximum_lat,
    ) = thailand.total_bounds

    padding = 0.03

    minimum_lon -= padding
    maximum_lon += padding
    minimum_lat -= padding
    maximum_lat += padding

    # ลดความละเอียดสำหรับหน้าเว็บ เพื่อให้ Render Free ประมวลผลเร็วขึ้น
    grid_width = 150
    grid_height = 250

    longitude_values = np.linspace(
        minimum_lon,
        maximum_lon,
        grid_width,
    )

    latitude_values = np.linspace(
        minimum_lat,
        maximum_lat,
        grid_height,
    )

    grid_lon, grid_lat = np.meshgrid(
        longitude_values,
        latitude_values,
    )

    thailand_mask = create_thailand_mask(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        thailand=thailand,
    )

    image_bounds = [
        [minimum_lat, minimum_lon],
        [maximum_lat, maximum_lon],
    ]

    layer_definitions = {
        "tmin": {
            "source": today_data,
            "field": "tmin",
            "name": "WeatherToday - Tmin",
            "short_name": "Tmin",
            "unit": "°C",
            "cmap": "turbo",
            "source_name": "WeatherToday",
        },
        "tmax": {
            "source": today_data,
            "field": "tmax",
            "name": "WeatherToday - Tmax",
            "short_name": "Tmax",
            "unit": "°C",
            "cmap": "turbo",
            "source_name": "WeatherToday",
        },
        "temperature": {
            "source": today_data,
            "field": "temperature",
            "name": "WeatherToday - Current Temperature",
            "short_name": "Temperature",
            "unit": "°C",
            "cmap": "turbo",
            "source_name": "WeatherToday",
        },
        "rainfall": {
            "source": today_data,
            "field": "rainfall",
            "name": "WeatherToday - Daily Rainfall",
            "short_name": "Rainfall",
            "unit": "mm",
            "cmap": "Blues",
            "source_name": "WeatherToday",
        },
        "air_temperature_3h": {
            "source": weather3h_data,
            "field": "air_temperature_3h",
            "name": "Weather3Hours - Air Temperature",
            "short_name": "Air Temperature",
            "unit": "°C",
            "cmap": "turbo",
            "source_name": "Weather3Hours",
        },
        "rainfall_3h": {
            "source": weather3h_data,
            "field": "rainfall_3h",
            "name": "Weather3Hours - Rainfall",
            "short_name": "Rainfall",
            "unit": "mm",
            "cmap": "Blues",
            "source_name": "Weather3Hours",
        },
        "rainfall_24h_3h": {
            "source": weather3h_data,
            "field": "rainfall_24h_3h",
            "name": "Weather3Hours - Rainfall 24 Hour",
            "short_name": "Rainfall 24h",
            "unit": "mm",
            "cmap": "Blues",
            "source_name": "Weather3Hours",
        },
    }

    overlays: dict[str, dict] = {}

    for (
        layer_key,
        definition,
    ) in layer_definitions.items():

        source_data = definition[
            "source"
        ]

        field = definition[
            "field"
        ]

        valid_data = source_data.dropna(
            subset=[
                "longitude",
                "latitude",
                field,
            ]
        )

        if valid_data.empty:
            continue

        station_lon = valid_data[
            "longitude"
        ].to_numpy()

        station_lat = valid_data[
            "latitude"
        ].to_numpy()

        station_values = valid_data[
            field
        ].to_numpy()

        grid_values = idw_interpolation(
            station_lon=station_lon,
            station_lat=station_lat,
            station_value=station_values,
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            power=2.0,
            nearest_points=8,
        )

        masked_values = np.ma.masked_where(
            ~thailand_mask,
            grid_values,
        )

        is_rainfall = (
            definition["unit"] == "mm"
        )

        if is_rainfall:
            minimum_value = 0.0
            maximum_value = float(
                np.ceil(
                    station_values.max()
                )
            )
        else:
            minimum_value = float(
                np.floor(
                    station_values.min()
                )
            )

            maximum_value = float(
                np.ceil(
                    station_values.max()
                )
            )

        if (
            minimum_value
            == maximum_value
        ):
            maximum_value += 1.0

        levels = np.linspace(
            minimum_value,
            maximum_value,
            27,
        )

        figure, axis = plt.subplots(
            figsize=(8, 12),
            dpi=110,
        )

        axis.contourf(
            grid_lon,
            grid_lat,
            masked_values,
            levels=levels,
            cmap=definition["cmap"],
            vmin=minimum_value,
            vmax=maximum_value,
            extend=(
                "max"
                if is_rainfall
                else "both"
            ),
            antialiased=True,
        )

        axis.set_xlim(
            minimum_lon,
            maximum_lon,
        )

        axis.set_ylim(
            minimum_lat,
            maximum_lat,
        )

        axis.axis("off")

        figure.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1,
        )

        image_buffer = io.BytesIO()

        figure.savefig(
            image_buffer,
            format="png",
            transparent=True,
            bbox_inches=None,
            pad_inches=0,
        )

        plt.close(
            figure
        )

        image_buffer.seek(
            0
        )

        encoded_image = base64.b64encode(
            image_buffer.read()
        ).decode(
            "utf-8"
        )

        overlays[layer_key] = {
            "name": definition["name"],
            "short_name": (
                definition["short_name"]
            ),
            "source_name": (
                definition["source_name"]
            ),
            "unit": definition["unit"],
            "image_url": (
                "data:image/png;base64,"
                + encoded_image
            ),
            "bounds": image_bounds,
            "minimum": minimum_value,
            "maximum": maximum_value,
        }

    if not overlays:
        raise ValueError(
            "ไม่พบข้อมูลที่สามารถสร้าง Layer ได้"
        )

    return (
        overlays,
        today_data,
        weather3h_data,
        thailand,
        today_error,
        weather3h_error,
    )



def create_weather_overlays(
    force_refresh: bool = False,
):
    """
    คืนค่า Layer แผนที่จาก Cache

    - Cache 30 นาที เพื่อลดการเรียก TMD API และการทำ IDW ซ้ำ
    - ใช้ Lock ป้องกันหลายคำขอสร้างแผนที่พร้อมกัน
    - หากการอัปเดตล้มเหลว แต่มี Cache เดิม จะใช้ Cache เดิมต่อ
    """

    current_time = time.time()
    cached_value = weather_overlay_cache.get("value")
    cache_age = (
        current_time
        - float(weather_overlay_cache.get("created_at", 0.0))
    )

    if (
        not force_refresh
        and cached_value is not None
        and cache_age < WEATHER_OVERLAY_CACHE_SECONDS
    ):
        return cached_value

    with weather_overlay_cache_lock:
        current_time = time.time()
        cached_value = weather_overlay_cache.get("value")
        cache_age = (
            current_time
            - float(weather_overlay_cache.get("created_at", 0.0))
        )

        # ตรวจซ้ำหลังได้ Lock เผื่อคำขอก่อนหน้าสร้างเสร็จแล้ว
        if (
            not force_refresh
            and cached_value is not None
            and cache_age < WEATHER_OVERLAY_CACHE_SECONDS
        ):
            return cached_value

        try:
            fresh_value = build_weather_overlays()
        except Exception:
            # หาก TMD API มีปัญหาชั่วคราว ให้หน้าเว็บยังใช้ Cache เก่าได้
            if cached_value is not None:
                return cached_value
            raise

        weather_overlay_cache["value"] = fresh_value
        weather_overlay_cache["created_at"] = time.time()

        return fresh_value



def create_publication_map(
    layer_key: str,
) -> tuple[io.BytesIO, str]:
    ensure_plot_imports()
    """
    สร้างแผนที่ประเทศไทยสำหรับรายงาน/งานวิชาการ
    ตาม Layer ที่ผู้ใช้เลือก และส่งออกเป็น PNG 300 DPI
    """

    thailand = load_thailand_boundary()

    today_layer_keys = {
        "tmin",
        "tmax",
        "temperature",
        "rainfall",
    }

    weather3h_layer_keys = {
        "air_temperature_3h",
        "rainfall_3h",
        "rainfall_24h_3h",
    }

    if layer_key in today_layer_keys:
        today_data = load_station_data()
        weather3h_data = pd.DataFrame()

    elif layer_key in weather3h_layer_keys:
        today_data = pd.DataFrame()
        weather3h_data = load_weather3hours_data()

    else:
        raise ValueError(
            f"ไม่รองรับ Layer: {layer_key}"
        )

    layer_definitions = {
        "tmin": {
            "data": today_data,
            "field": "tmin",
            "title": "Thailand Minimum Temperature Map",
            "subtitle": "WeatherToday",
            "unit": "°C",
            "cmap": "turbo",
            "time_field": "observation_datetime",
            "file_name": "Thailand_Minimum_Temperature.png",
        },
        "tmax": {
            "data": today_data,
            "field": "tmax",
            "title": "Thailand Maximum Temperature Map",
            "subtitle": "WeatherToday",
            "unit": "°C",
            "cmap": "turbo",
            "time_field": "observation_datetime",
            "file_name": "Thailand_Maximum_Temperature.png",
        },
        "temperature": {
            "data": today_data,
            "field": "temperature",
            "title": "Thailand Current Temperature Map",
            "subtitle": "WeatherToday",
            "unit": "°C",
            "cmap": "turbo",
            "time_field": "observation_datetime",
            "file_name": "Thailand_Current_Temperature.png",
        },
        "rainfall": {
            "data": today_data,
            "field": "rainfall",
            "title": "Thailand Daily Rainfall Map",
            "subtitle": "WeatherToday",
            "unit": "mm",
            "cmap": "Blues",
            "time_field": "observation_datetime",
            "file_name": "Thailand_Daily_Rainfall.png",
        },
        "air_temperature_3h": {
            "data": weather3h_data,
            "field": "air_temperature_3h",
            "title": "Thailand Air Temperature Map",
            "subtitle": "Weather3Hours",
            "unit": "°C",
            "cmap": "turbo",
            "time_field": "observation_datetime_3h",
            "file_name": "Thailand_Air_Temperature_3Hours.png",
        },
        "rainfall_3h": {
            "data": weather3h_data,
            "field": "rainfall_3h",
            "title": "Thailand 3-Hour Rainfall Map",
            "subtitle": "Weather3Hours",
            "unit": "mm",
            "cmap": "Blues",
            "time_field": "observation_datetime_3h",
            "file_name": "Thailand_Rainfall_3Hours.png",
        },
        "rainfall_24h_3h": {
            "data": weather3h_data,
            "field": "rainfall_24h_3h",
            "title": "Thailand 24-Hour Rainfall Map",
            "subtitle": "Weather3Hours",
            "unit": "mm",
            "cmap": "Blues",
            "time_field": "observation_datetime_3h",
            "file_name": "Thailand_Rainfall_24Hours.png",
        },
    }

    definition = layer_definitions.get(
        layer_key
    )

    if definition is None:
        raise ValueError(
            f"ไม่รองรับ Layer: {layer_key}"
        )

    source_data = definition["data"]

    valid_data = source_data.dropna(
        subset=[
            "longitude",
            "latitude",
            definition["field"],
        ]
    ).copy()

    if valid_data.empty:
        raise ValueError(
            "Layer ที่เลือกไม่มีข้อมูลสำหรับสร้างแผนที่"
        )

    (
        minimum_lon,
        minimum_lat,
        maximum_lon,
        maximum_lat,
    ) = thailand.total_bounds

    padding = 0.20

    minimum_lon -= padding
    maximum_lon += padding
    minimum_lat -= padding
    maximum_lat += padding

    grid_width = 500
    grid_height = 780

    longitude_values = np.linspace(
        minimum_lon,
        maximum_lon,
        grid_width,
    )

    latitude_values = np.linspace(
        minimum_lat,
        maximum_lat,
        grid_height,
    )

    grid_lon, grid_lat = np.meshgrid(
        longitude_values,
        latitude_values,
    )

    grid_values = idw_interpolation(
        station_lon=valid_data[
            "longitude"
        ].to_numpy(),
        station_lat=valid_data[
            "latitude"
        ].to_numpy(),
        station_value=valid_data[
            definition["field"]
        ].to_numpy(),
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        power=2.0,
        nearest_points=8,
    )

    thailand_mask = create_thailand_mask(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        thailand=thailand,
    )

    masked_values = np.ma.masked_where(
        ~thailand_mask,
        grid_values,
    )

    station_values = valid_data[
        definition["field"]
    ].to_numpy()

    if definition["unit"] == "mm":
        minimum_value = 0.0
        maximum_value = float(
            np.ceil(
                station_values.max()
            )
        )
    else:
        minimum_value = float(
            np.floor(
                station_values.min()
            )
        )
        maximum_value = float(
            np.ceil(
                station_values.max()
            )
        )

    if minimum_value == maximum_value:
        maximum_value += 1.0

    levels = np.linspace(
        minimum_value,
        maximum_value,
        27,
    )

    figure = plt.figure(
        figsize=(8.27, 11.69),
        dpi=300,
        facecolor="white",
    )

    map_axis = figure.add_axes(
        [0.10, 0.18, 0.80, 0.68]
    )

    contour = map_axis.contourf(
        grid_lon,
        grid_lat,
        masked_values,
        levels=levels,
        cmap=definition["cmap"],
        vmin=minimum_value,
        vmax=maximum_value,
        extend=(
            "max"
            if definition["unit"] == "mm"
            else "both"
        ),
        antialiased=True,
        zorder=1,
    )

    thailand.boundary.plot(
        ax=map_axis,
        color="black",
        linewidth=0.65,
        zorder=3,
    )

    map_axis.scatter(
        valid_data["longitude"],
        valid_data["latitude"],
        s=10,
        facecolor="black",
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
        label="Weather station",
    )

    map_axis.set_xlim(
        minimum_lon,
        maximum_lon,
    )

    map_axis.set_ylim(
        minimum_lat,
        maximum_lat,
    )

    map_axis.set_aspect(
        "equal",
        adjustable="box",
    )

    map_axis.axis("off")

    # North arrow
    map_axis.annotate(
        "N",
        xy=(0.92, 0.93),
        xytext=(0.92, 0.83),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        arrowprops={
            "facecolor": "black",
            "edgecolor": "black",
            "width": 2.0,
            "headwidth": 8.0,
            "headlength": 10.0,
        },
        zorder=10,
    )

    # Approximate 200-km scale bar
    scale_latitude = minimum_lat + 0.75
    scale_start_lon = minimum_lon + 0.75

    scale_length_km = 200.0

    longitude_degrees = (
        scale_length_km
        /
        (
            111.32
            *
            np.cos(
                np.deg2rad(
                    scale_latitude
                )
            )
        )
    )

    scale_end_lon = (
        scale_start_lon
        + longitude_degrees
    )

    map_axis.plot(
        [
            scale_start_lon,
            scale_end_lon,
        ],
        [
            scale_latitude,
            scale_latitude,
        ],
        color="black",
        linewidth=2.5,
        zorder=10,
    )

    map_axis.plot(
        [
            scale_start_lon,
            scale_start_lon,
        ],
        [
            scale_latitude - 0.08,
            scale_latitude + 0.08,
        ],
        color="black",
        linewidth=1.8,
        zorder=10,
    )

    map_axis.plot(
        [
            scale_end_lon,
            scale_end_lon,
        ],
        [
            scale_latitude - 0.08,
            scale_latitude + 0.08,
        ],
        color="black",
        linewidth=1.8,
        zorder=10,
    )

    map_axis.text(
        (
            scale_start_lon
            + scale_end_lon
        )
        / 2,
        scale_latitude + 0.16,
        "200 km",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=10,
    )

    observation_times = (
        valid_data[
            definition["time_field"]
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    observation_times = observation_times[
        observation_times != ""
    ]

    observation_time = (
        format_observation_datetime(
            observation_times.max()
        )
        if not observation_times.empty
        else "Unknown"
    )

    figure.text(
        0.50,
        0.945,
        definition["title"],
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    figure.text(
        0.50,
        0.915,
        (
            f'{definition["subtitle"]} | '
            f'Observation time: '
            f'{observation_time}'
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    figure.text(
        0.50,
        0.892,
        (
            "Interpolation: Inverse Distance "
            "Weighting (IDW)"
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    colorbar_axis = figure.add_axes(
        [0.16, 0.115, 0.68, 0.025]
    )

    colorbar = figure.colorbar(
        contour,
        cax=colorbar_axis,
        orientation="horizontal",
        format=FormatStrFormatter("%.1f"),
    )

    colorbar.set_label(
        definition["unit"],
        fontsize=10,
        fontweight="bold",
    )

    colorbar.ax.tick_params(
        labelsize=8
    )

    colorbar.formatter = FormatStrFormatter(
        "%.1f"
    )

    colorbar.update_ticks()


    figure.text(
        0.10,
        0.075,
        (
            "● Weather station"
        ),
        ha="left",
        va="center",
        fontsize=8,
    )

    figure.text(
        0.10,
        0.050,
        (
            "Source: Thai Meteorological Department"
        ),
        ha="left",
        va="center",
        fontsize=8,
    )

    figure.text(
        0.10,
        0.030,
        (
            "Coordinate Reference System: WGS 84 "
            "(EPSG:4326)"
        ),
        ha="left",
        va="center",
        fontsize=8,
    )

    figure.text(
        0.90,
        0.030,
        (
            f"Stations: {len(valid_data)}"
        ),
        ha="right",
        va="center",
        fontsize=8,
    )

    output_buffer = io.BytesIO()

    figure.savefig(
        output_buffer,
        format="png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(
        figure
    )

    output_buffer.seek(
        0
    )

    return (
        output_buffer,
        definition["file_name"],
    )


@app.get(
    "/export/publication",
)
def export_publication_map(
    layer: str = Query(
        default="tmin"
    ),
) -> StreamingResponse:
    """
    ส่งออกแผนที่แบบเป็นทางการ PNG 300 DPI
    ตาม Layer ที่เลือกบนหน้าเว็บ
    """

    try:
        image_buffer, file_name = (
            create_publication_map(
                layer_key=layer
            )
        )

        return StreamingResponse(
            image_buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{file_name}"'
                )
            },
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


# =========================================================
# 10. ค้นหาสถานที่ผ่าน Nominatim
# =========================================================

def clean_search_cache() -> None:
    """
    ลบ Cache ที่หมดอายุ
    """

    current_time = time.time()

    expired_keys = [
        key
        for key, item in search_cache.items()
        if (
            current_time - item["created_at"]
            > CACHE_SECONDS
        )
    ]

    for key in expired_keys:
        search_cache.pop(
            key,
            None,
        )

    if len(search_cache) > MAX_CACHE_ITEMS:
        oldest_keys = sorted(
            search_cache,
            key=lambda key: search_cache[key][
                "created_at"
            ],
        )

        remove_count = (
            len(search_cache)
            - MAX_CACHE_ITEMS
        )

        for key in oldest_keys[:remove_count]:
            search_cache.pop(
                key,
                None,
            )


def call_nominatim(
    search_text: str,
) -> list[dict]:
    """
    เรียก Nominatim โดยเว้นระยะอย่างน้อย 1 วินาที
    และจำกัดผลลัพธ์เฉพาะประเทศไทย
    """

    global last_nominatim_request_time

    normalized_query = (
        search_text.strip().lower()
    )

    clean_search_cache()

    cached_result = search_cache.get(
        normalized_query
    )

    if cached_result is not None:
        return cached_result["results"]

    with nominatim_lock:
        elapsed = (
            time.monotonic()
            - last_nominatim_request_time
        )

        if elapsed < 1.1:
            time.sleep(
                1.1 - elapsed
            )

        headers = {
            "User-Agent": NOMINATIM_USER_AGENT,
            "Accept-Language": (
                "th,en;q=0.8"
            ),
        }

        parameters = {
            "q": search_text,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 8,
            "countrycodes": "th",
            "accept-language": "th,en",
            "dedupe": 1,
        }

        try:
            with httpx.Client(
                timeout=15.0,
                follow_redirects=True,
            ) as client:
                response = client.get(
                    NOMINATIM_SEARCH_URL,
                    params=parameters,
                    headers=headers,
                )

            last_nominatim_request_time = (
                time.monotonic()
            )

            response.raise_for_status()

            raw_results = response.json()

        except httpx.TimeoutException as error:
            raise RuntimeError(
                "Nominatim ใช้เวลาตอบกลับนานเกินไป"
            ) from error

        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                "Nominatim ตอบกลับด้วยข้อผิดพลาด "
                f"{error.response.status_code}"
            ) from error

        except httpx.HTTPError as error:
            raise RuntimeError(
                "ไม่สามารถเชื่อมต่อบริการค้นหาสถานที่ได้"
            ) from error

    cleaned_results: list[dict] = []

    for item in raw_results:
        try:
            latitude = float(
                item["lat"]
            )

            longitude = float(
                item["lon"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        bounding_box = item.get(
            "boundingbox"
        )

        cleaned_bounding_box = None

        if (
            isinstance(bounding_box, list)
            and len(bounding_box) == 4
        ):
            try:
                cleaned_bounding_box = [
                    float(bounding_box[0]),
                    float(bounding_box[1]),
                    float(bounding_box[2]),
                    float(bounding_box[3]),
                ]
            except (
                TypeError,
                ValueError,
            ):
                cleaned_bounding_box = None

        cleaned_results.append(
            {
                "display_name": item.get(
                    "display_name",
                    "ไม่ทราบชื่อสถานที่",
                ),
                "latitude": latitude,
                "longitude": longitude,
                "boundingbox": cleaned_bounding_box,
                "type": item.get(
                    "type",
                    "",
                ),
                "category": item.get(
                    "category",
                    item.get("class", ""),
                ),
            }
        )

    search_cache[normalized_query] = {
        "created_at": time.time(),
        "results": cleaned_results,
    }

    return cleaned_results


@app.get(
    "/api/search",
    response_class=JSONResponse,
)
def search_place(
    q: str = Query(
        ...,
        min_length=2,
        max_length=150,
    ),
) -> JSONResponse:
    """
    API สำหรับค้นหาสถานที่ในประเทศไทย
    """

    search_text = q.strip()

    if len(search_text) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "กรุณาพิมพ์อย่างน้อย 2 ตัวอักษร"
            ),
        )

    try:
        results = call_nominatim(
            search_text
        )

        return JSONResponse(
            content={
                "query": search_text,
                "results": results,
                "attribution": (
                    "Search data © OpenStreetMap contributors"
                ),
            }
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error


# =========================================================
# 11. หน้าแรก
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home() -> HTMLResponse:
    """
    หน้าแรก
    """

    page_html = """
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <title>Thailand Tmin Map</title>

        <style>
            body {
                margin: 0;
                background: #edf2f6;
                font-family: Arial, sans-serif;
                text-align: center;
            }

            .header {
                padding: 22px;
                background: #173f6d;
                color: white;
            }

            .header h1 {
                margin: 0;
            }

            .content {
                padding: 30px;
            }

            .button {
                display: inline-block;
                margin: 7px;
                padding: 12px 20px;
                border-radius: 6px;
                background: #2874b2;
                color: white;
                text-decoration: none;
                font-weight: bold;
            }

            .button:hover {
                background: #155582;
            }
        </style>
    </head>

    <body>
        <div class="header">
            <h1>
                Thailand Weather Interpolation Map
            </h1>

            <p>
                แผนที่อากาศแบบเลือก Layer จาก TMD API
            </p>
        </div>

        <div class="content">
            <a class="button" href="/map">
                เปิดแผนที่
            </a>

            <a class="button" href="/stations">
                ดูข้อมูลสถานี
            </a>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(
        content=page_html
    )


# =========================================================
# 12. หน้าแผนที่แบบ Lazy Layer
# =========================================================

def _get_boundary_payload() -> tuple[dict, list[list[float]], object]:
    """Return cached Thailand GeoJSON, Leaflet bounds, and GeoDataFrame."""
    ensure_data_imports()
    now = time.time()
    cached_geojson = boundary_cache.get("geojson")
    cached_bounds = boundary_cache.get("bounds")
    cached_thailand = boundary_cache.get("thailand")
    created_at = float(boundary_cache.get("created_at", 0.0))

    if (
        cached_geojson is not None
        and cached_bounds is not None
        and cached_thailand is not None
        and now - created_at < LAYER_CACHE_SECONDS
    ):
        return cached_geojson, cached_bounds, cached_thailand

    thailand = load_thailand_boundary()
    min_lon, min_lat, max_lon, max_lat = thailand.total_bounds
    geojson = json.loads(thailand.to_json())
    bounds = [[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]]

    boundary_cache.update(
        {
            "created_at": now,
            "geojson": geojson,
            "bounds": bounds,
            "thailand": thailand,
        }
    )
    return geojson, bounds, thailand


def _interpolate_rgb(normalized: object, palette: str) -> object:
    """Convert normalized values (0..1) to RGB without importing Matplotlib."""
    ensure_data_imports()

    if palette == "blues":
        stops = np.array(
            [
                [247, 251, 255], [222, 235, 247], [198, 219, 239],
                [158, 202, 225], [107, 174, 214], [66, 146, 198],
                [33, 113, 181], [8, 81, 156], [8, 48, 107],
            ],
            dtype=float,
        )
    else:
        stops = np.array(
            [
                [48, 18, 59], [65, 69, 171], [70, 117, 237],
                [57, 162, 252], [27, 207, 212], [36, 236, 166],
                [97, 252, 108], [164, 252, 60], [209, 232, 52],
                [249, 186, 56], [246, 107, 25], [217, 56, 6],
                [122, 4, 3],
            ],
            dtype=float,
        )

    position = np.clip(normalized, 0.0, 1.0) * (len(stops) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(stops) - 1)
    fraction = (position - low)[..., None]
    rgb = stops[low] * (1.0 - fraction) + stops[high] * fraction
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _build_single_layer(layer_key: str) -> dict:
    """Generate only the requested web overlay and station payload."""
    ensure_data_imports()
    from PIL import Image

    definition = LAYER_DEFINITIONS.get(layer_key)
    if definition is None:
        raise ValueError(f"ไม่รองรับ Layer: {layer_key}")

    if definition["source"] == "today":
        source_data = load_station_data()
        time_field = "observation_datetime"
    else:
        source_data = load_weather3hours_data()
        time_field = "observation_datetime_3h"

    field = definition["field"]
    valid_data = source_data.dropna(
        subset=["longitude", "latitude", field]
    ).copy()
    if valid_data.empty:
        raise ValueError("Layer ที่เลือกไม่มีข้อมูลสำหรับสร้างแผนที่")

    _, boundary_bounds, thailand = _get_boundary_payload()
    min_lon, min_lat, max_lon, max_lat = thailand.total_bounds
    padding = 0.03
    min_lon -= padding
    max_lon += padding
    min_lat -= padding
    max_lat += padding

    # A compact grid is sufficient for an interactive web overlay and is
    # much safer on Render Free than creating all seven high-resolution maps.
    grid_width = 120
    grid_height = 200
    lon_values = np.linspace(min_lon, max_lon, grid_width)
    lat_values = np.linspace(min_lat, max_lat, grid_height)
    grid_lon, grid_lat = np.meshgrid(lon_values, lat_values)

    values = valid_data[field].to_numpy(dtype=float)
    grid_values = idw_interpolation(
        station_lon=valid_data["longitude"].to_numpy(dtype=float),
        station_lat=valid_data["latitude"].to_numpy(dtype=float),
        station_value=values,
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        power=2.0,
        nearest_points=8,
    )
    thailand_mask = create_thailand_mask(grid_lon, grid_lat, thailand)

    if definition["unit"] == "mm":
        minimum = 0.0
        maximum = float(np.ceil(values.max()))
    else:
        minimum = float(np.floor(values.min()))
        maximum = float(np.ceil(values.max()))
    if maximum <= minimum:
        maximum = minimum + 1.0

    normalized = (grid_values - minimum) / (maximum - minimum)
    rgb = _interpolate_rgb(normalized, definition["palette"])
    alpha = np.where(thailand_mask, 205, 0).astype(np.uint8)
    rgba = np.dstack((rgb, alpha))

    # Latitude grid is south-to-north; PNG rows are top-to-bottom.
    rgba = np.flipud(rgba)
    image = Image.fromarray(rgba, mode="RGBA")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG", optimize=True)
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    station_columns = ["station", "province", "latitude", "longitude", field]
    station_payload = []
    for record in valid_data[station_columns].to_dict(orient="records"):
        station_payload.append(
            {
                "station": str(record.get("station", "Weather Station")),
                "province": str(record.get("province", "")),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "value": float(record[field]),
            }
        )

    times = valid_data[time_field].dropna().astype(str).str.strip()
    times = times[times != ""]
    observation_time = (
        format_observation_datetime(times.max()) if not times.empty else "Unknown"
    )

    return {
        "layer": layer_key,
        "name": definition["name"],
        "short_name": definition["short_name"],
        "unit": definition["unit"],
        "minimum": minimum,
        "maximum": maximum,
        "observation_time": observation_time,
        "image_url": "data:image/png;base64," + encoded_image,
        "bounds": [[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]],
        "boundary_bounds": boundary_bounds,
        "stations": station_payload,
        "cached_at": time.time(),
    }


def get_single_layer(layer_key: str, force_refresh: bool = False) -> dict:
    """Return one cached layer; never build all layers in a single request."""
    now = time.time()
    cached = layer_cache.get(layer_key)
    if (
        not force_refresh
        and cached is not None
        and now - float(cached.get("cached_at", 0.0)) < LAYER_CACHE_SECONDS
    ):
        return cached

    with layer_cache_lock:
        now = time.time()
        cached = layer_cache.get(layer_key)
        if (
            not force_refresh
            and cached is not None
            and now - float(cached.get("cached_at", 0.0)) < LAYER_CACHE_SECONDS
        ):
            return cached

        try:
            fresh = _build_single_layer(layer_key)
        except Exception:
            if cached is not None:
                return cached
            raise

        layer_cache[layer_key] = fresh
        return fresh


@app.get("/api/boundary", response_class=JSONResponse)
def api_boundary() -> JSONResponse:
    try:
        geojson, bounds, _ = _get_boundary_payload()
        return JSONResponse({"geojson": geojson, "bounds": bounds})
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/overlay", response_class=JSONResponse)
def api_overlay(
    layer: str = Query(default="tmin"),
    refresh: bool = Query(default=False),
) -> JSONResponse:
    try:
        return JSONResponse(get_single_layer(layer, force_refresh=refresh))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/map", response_class=HTMLResponse)
def show_map() -> HTMLResponse:
    """Return a lightweight page immediately; layers load one at a time."""
    layer_options = "".join(
        f'<option value="{key}">{html.escape(value["name"])}</option>'
        for key, value in LAYER_DEFINITIONS.items()
    )

    page = r"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Thailand Weather Interpolation Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map { height: 100%; margin: 0; }
    body { font-family: Arial, sans-serif; }
    #panel { position: fixed; z-index: 1000; top: 12px; left: 50px; width: min(390px, calc(100vw - 80px)); background: rgba(255,255,255,.97); padding: 11px; border-radius: 8px; box-shadow: 0 2px 9px rgba(0,0,0,.30); }
    #panel h2 { margin: 0 0 8px; font-size: 16px; }
    .row { display: flex; gap: 7px; margin-top: 7px; }
    select, input, button { box-sizing: border-box; padding: 9px; border: 1px solid #aaa; border-radius: 5px; font-size: 14px; }
    select, input { flex: 1; min-width: 0; }
    button { color: white; background: #1769aa; border: 0; cursor: pointer; }
    button:disabled { background: #777; cursor: wait; }
    #export-button { background: #176b45; }
    #status { margin-top: 8px; padding: 7px; border-radius: 5px; background: #f1f3f5; font-size: 13px; }
    #search-results { display: none; max-height: 220px; overflow-y: auto; border: 1px solid #ccc; margin-top: 6px; background: white; }
    .result { padding: 8px; border-bottom: 1px solid #eee; cursor: pointer; font-size: 13px; }
    .result:hover { background: #eaf4ff; }
    #legend { position: fixed; z-index: 1000; left: 35px; bottom: 35px; width: 220px; background: rgba(255,255,255,.96); padding: 12px; border: 2px solid #555; border-radius: 7px; }
    #gradient { height: 16px; margin: 8px 0 5px; }
    #range { display: flex; justify-content: space-between; font-size: 12px; }
    #value-box { position: fixed; z-index: 1000; right: 35px; bottom: 35px; min-width: 220px; background: rgba(255,255,255,.96); padding: 12px; border: 2px solid #555; border-radius: 7px; line-height: 1.45; }
    #busy { display:none; position: fixed; z-index: 2000; inset: 0; background: rgba(255,255,255,.75); align-items:center; justify-content:center; font-size: 20px; font-weight:bold; }
    @media(max-width:700px){ #panel{left:10px;top:10px;width:calc(100vw - 40px)} #legend{left:10px;bottom:10px;width:165px} #value-box{right:10px;bottom:10px;min-width:145px;font-size:12px} }
  </style>
</head>
<body>
<div id="map"></div>
<div id="busy">กำลังสร้าง Layer กรุณารอสักครู่…</div>
<div id="panel">
  <h2>Thailand Weather Interpolation Map</h2>
  <div class="row"><select id="layer-select">__LAYER_OPTIONS__</select><button id="load-button">แสดง Layer</button></div>
  <div class="row"><input id="search-input" placeholder="ค้นหาจังหวัด อำเภอ ตำบล หรือสถานที่"><button id="search-button">ค้นหา</button></div>
  <div class="row"><button id="export-button">Export Publication Map</button><button id="station-button">ซ่อนสถานี</button></div>
  <div id="status">กำลังเตรียมแผนที่พื้นฐาน…</div>
  <div id="search-results"></div>
</div>
<div id="legend"><b id="legend-title">Layer</b><div id="gradient"></div><div id="range"><span id="min-value">-</span><span id="max-value">-</span></div></div>
<div id="value-box"><b id="value-title">Interpolated value</b><br>เลื่อนเมาส์ภายในประเทศไทย</div>
<script>
const map = L.map('map', {preferCanvas:true}).setView([13,101],5);
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:18, attribution:'© OpenStreetMap contributors'}).addTo(map);
const light = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {maxZoom:19, attribution:'© OpenStreetMap contributors © CARTO'});
const dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {maxZoom:19, attribution:'© OpenStreetMap contributors © CARTO'});
L.control.layers({'OpenStreetMap': osm, 'พื้นอ่อน':light, 'พื้นเข้ม':dark}, {}, {collapsed:true}).addTo(map);
let overlay=null, boundary=null, stationGroup=L.layerGroup().addTo(map), searchMarker=null;
let currentData=null, stationsVisible=true;
const statusBox=document.getElementById('status'), busy=document.getElementById('busy');
function setBusy(on,msg){busy.style.display=on?'flex':'none'; if(msg) busy.textContent=msg; document.getElementById('load-button').disabled=on;}
function gradient(unit){return unit==='mm'?'linear-gradient(to right,#f7fbff,#deebf7,#c6dbef,#9ecae1,#6baed6,#4292c6,#2171b5,#08519c,#08306b)':'linear-gradient(to right,#30123b,#4145ab,#4675ed,#39a2fc,#1bcfd4,#24eca6,#61fc6c,#a4fc3c,#d1e834,#f9ba38,#f66b19,#d93806,#7a0403)';}
function updateLegend(d){document.getElementById('legend-title').textContent=d.name;document.getElementById('gradient').style.background=gradient(d.unit);document.getElementById('min-value').textContent=Number(d.minimum).toFixed(1)+' '+d.unit;document.getElementById('max-value').textContent=Number(d.maximum).toFixed(1)+' '+d.unit;const valueTitle = document.getElementById('value-title');
if (valueTitle) {
    valueTitle.textContent = 'Interpolated '+d.short_name;
}}
function drawStations(d){stationGroup.clearLayers(); for(const s of d.stations){const marker=L.circleMarker([s.latitude,s.longitude],{radius:4,color:'#fff',weight:1,fillColor:'#111',fillOpacity:.9});marker.bindPopup(`<b>${String(s.station).replaceAll('<','&lt;')}</b><br>Province: ${String(s.province).replaceAll('<','&lt;')}<br>${d.short_name}: ${Number(s.value).toFixed(1)} ${d.unit}<br>Lat: ${s.latitude.toFixed(4)}<br>Lon: ${s.longitude.toFixed(4)}`);marker.addTo(stationGroup);} if(!stationsVisible) map.removeLayer(stationGroup);}
async function loadLayer(force=false){const key=document.getElementById('layer-select').value;setBusy(true,'กำลังสร้าง '+key+'…');statusBox.textContent='กำลังดึงข้อมูลและสร้าง Layer เฉพาะรายการที่เลือก';try{const r=await fetch('/api/overlay?layer='+encodeURIComponent(key)+(force?'&refresh=true':''));const d=await r.json();if(!r.ok) throw new Error(d.detail||'โหลด Layer ไม่สำเร็จ');currentData=d;if(overlay) map.removeLayer(overlay);overlay=L.imageOverlay(d.image_url,d.bounds,{opacity:.74,interactive:false}).addTo(map);overlay.bringToFront();drawStations(d);updateLegend(d);statusBox.textContent=d.name+' | Observation: '+d.observation_time+' | Stations: '+d.stations.length;map.fitBounds(d.boundary_bounds);}catch(e){statusBox.textContent='Error: '+e.message;alert(e.message);}finally{setBusy(false);}}
fetch('/api/boundary').then(r=>r.json()).then(d=>{boundary=L.geoJSON(d.geojson,{style:{color:'#111',weight:1.2,fillOpacity:0}}).addTo(map);map.fitBounds(d.bounds);statusBox.textContent='แผนที่พื้นฐานพร้อมแล้ว';loadLayer(false);}).catch(e=>{statusBox.textContent='โหลดขอบเขตประเทศไทยไม่สำเร็จ: '+e.message;});
document.getElementById('load-button').onclick=()=>loadLayer(false);
document.getElementById('export-button').onclick=()=>{const key=document.getElementById('layer-select').value;window.location.href='/export/publication?layer='+encodeURIComponent(key);};
document.getElementById('station-button').onclick=(event)=>{stationsVisible=!stationsVisible;if(stationsVisible){stationGroup.addTo(map);event.target.textContent='ซ่อนสถานี';}else{map.removeLayer(stationGroup);event.target.textContent='แสดงสถานี';}};
map.on('mousemove',e=>{if(!currentData||!currentData.stations.length)return;let nearest=currentData.stations.map(s=>({s,d:(s.latitude-e.latlng.lat)**2+(s.longitude-e.latlng.lng)**2})).sort((a,b)=>a.d-b.d).slice(0,8);let sw=0,sv=0;for(const x of nearest){const w=1/Math.max(x.d,1e-10);sw+=w;sv+=w*x.s.value;}const valueBox = document.getElementById('value-box');
if (valueBox) {
    valueBox.innerHTML =
        '<b id="value-title">Interpolated ' + currentData.short_name + '</b><br>' +
        '<b>Interpolated '+currentData.short_name+'</b><br>'+((sv/sw).toFixed(1))+' '+currentData.unit+'<br>Lat: '+e.latlng.lat.toFixed(4)+' | Lon: '+e.latlng.lng.toFixed(4);
}});


map.on('click', function (e) {

    if (!currentData || !currentData.stations || currentData.stations.length === 0) return;
    if (!boundary) return;

    const pointLongitude = e.latlng.lng;
    const pointLatitude = e.latlng.lat;

    function pointInRing(longitude, latitude, ring) {
        let inside = false;

        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
            const xi = ring[i][0];
            const yi = ring[i][1];
            const xj = ring[j][0];
            const yj = ring[j][1];

            const intersectsRing =
                ((yi > latitude) !== (yj > latitude)) &&
                (
                    longitude <
                    (xj - xi) * (latitude - yi) /
                    ((yj - yi) || Number.EPSILON) + xi
                );

            if (intersectsRing) {
                inside = !inside;
            }
        }

        return inside;
    }

    function pointInPolygon(longitude, latitude, polygonCoordinates) {
        if (!polygonCoordinates.length) return false;

        if (!pointInRing(longitude, latitude, polygonCoordinates[0])) {
            return false;
        }

        for (let i = 1; i < polygonCoordinates.length; i++) {
            if (pointInRing(longitude, latitude, polygonCoordinates[i])) {
                return false;
            }
        }

        return true;
    }

    function pointInsideThailand(longitude, latitude) {
        const geojson = boundary.toGeoJSON();
        const features = geojson.type === 'FeatureCollection'
            ? geojson.features
            : [geojson];

        for (const feature of features) {
            if (!feature || !feature.geometry) continue;

            const geometry = feature.geometry;

            if (
                geometry.type === 'Polygon' &&
                pointInPolygon(longitude, latitude, geometry.coordinates)
            ) {
                return true;
            }

            if (geometry.type === 'MultiPolygon') {
                for (const polygonCoordinates of geometry.coordinates) {
                    if (
                        pointInPolygon(
                            longitude,
                            latitude,
                            polygonCoordinates
                        )
                    ) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    if (!pointInsideThailand(pointLongitude, pointLatitude)) {
        map.closePopup();
        return;
    }

    const nearest = currentData.stations
        .map(s => ({
            station: s,
            d: Math.pow(s.latitude - e.latlng.lat, 2) +
               Math.pow(s.longitude - e.latlng.lng, 2)
        }))
        .sort((a, b) => a.d - b.d)
        .slice(0, 8);

    let value = 0;
    let weightSum = 0;

    nearest.forEach(item => {
        const w = 1 / Math.max(item.d, 1e-10);
        value += item.station.value * w;
        weightSum += w;
    });

    value /= weightSum;

    L.popup({
        closeButton:false,
        autoClose:true,
        closeOnClick:true,
        offset:[0,-5]
    })
    .setLatLng(e.latlng)
    .setContent(
        '<div style="font-size:22px;font-weight:bold;text-align:center;">'
        + value.toFixed(1) + ' ' + currentData.unit +
        '</div>'
    )
    .openOn(map);

});


async function searchPlace(){const q=document.getElementById('search-input').value.trim(),box=document.getElementById('search-results');if(q.length<2)return;document.getElementById('search-button').disabled=true;try{const r=await fetch('/api/search?q='+encodeURIComponent(q));const d=await r.json();if(!r.ok)throw new Error(d.detail||'ค้นหาไม่สำเร็จ');box.innerHTML='';for(const item of d.results){const div=document.createElement('div');div.className='result';div.textContent=item.display_name;div.onclick=()=>{if(searchMarker)map.removeLayer(searchMarker);searchMarker=L.marker([item.latitude,item.longitude]).addTo(map).bindPopup(item.display_name).openPopup();if(item.boundingbox){map.fitBounds([[item.boundingbox[0],item.boundingbox[2]],[item.boundingbox[1],item.boundingbox[3]]]);}else map.setView([item.latitude,item.longitude],13);box.style.display='none';};box.appendChild(div);}box.style.display=d.results.length?'block':'none';if(!d.results.length)statusBox.textContent='ไม่พบสถานที่';}catch(e){statusBox.textContent='Search error: '+e.message;}finally{document.getElementById('search-button').disabled=false;}}
document.getElementById('search-button').onclick=searchPlace;document.getElementById('search-input').addEventListener('keydown',e=>{if(e.key==='Enter')searchPlace();});
</script>
</body></html>"""
    return HTMLResponse(page.replace("__LAYER_OPTIONS__", layer_options))


# =========================================================
# 13. ตารางสถานี
# =========================================================

@app.get(
    "/stations",
    response_class=HTMLResponse,
)
def show_stations() -> HTMLResponse:
    """
    แสดงข้อมูลสถานี
    """

    try:
        station_data = load_station_data()

        table_html = station_data.to_html(
            index=False,
            border=0,
            classes="station-table",
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )

        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="th">
            <head>
                <meta charset="UTF-8">

                <meta
                    name="viewport"
                    content="
                        width=device-width,
                        initial-scale=1.0
                    "
                >

                <title>Station Data</title>

                <style>
                    body {{
                        margin: 0;
                        padding: 20px;
                        background: #edf2f6;
                        font-family: Arial, sans-serif;
                    }}

                    .container {{
                        max-width: 1000px;
                        margin: auto;
                        padding: 20px;
                        background: white;
                        border-radius: 8px;
                        overflow-x: auto;
                    }}

                    .station-table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}

                    .station-table th,
                    .station-table td {{
                        padding: 8px;
                        border: 1px solid #ccc;
                        text-align: center;
                    }}

                    .station-table th {{
                        color: white;
                        background: #173f6d;
                    }}

                    .station-table tr:nth-child(even) {{
                        background: #f2f6f9;
                    }}

                    .button {{
                        display: inline-block;
                        margin: 5px;
                        padding: 10px 16px;
                        border-radius: 6px;
                        background: #2874b2;
                        color: white;
                        text-decoration: none;
                    }}
                </style>
            </head>

            <body>
                <div class="container">
                    <a class="button" href="/">
                        กลับหน้าหลัก
                    </a>

                    <a class="button" href="/map">
                        เปิดแผนที่
                    </a>

                    <h2>
                        ข้อมูลสถานีตรวจอากาศ
                    </h2>

                    {table_html}
                </div>
            </body>
            </html>
            """
        )

    except Exception as error:
        safe_error = html.escape(
            str(error)
        )

        return HTMLResponse(
            content=f"""
            <h2>
                ไม่สามารถอ่านข้อมูลสถานีได้
            </h2>

            <pre>{safe_error}</pre>
            """,
            status_code=500,
        )
