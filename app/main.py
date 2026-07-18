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

import folium
import geopandas as gpd
import httpx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from folium.raster_layers import ImageOverlay
from scipy.spatial import cKDTree
from shapely import intersects, points


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

def create_weather_overlays():
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

    grid_width = 220
    grid_height = 380

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



def create_publication_map(
    layer_key: str,
) -> tuple[io.BytesIO, str]:
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
# 12. หน้าแผนที่
# =========================================================

@app.get(
    "/map",
    response_class=HTMLResponse,
)
def show_map() -> HTMLResponse:
    """
    หน้าแผนที่ Interactive
    """

    try:
        (
            weather_overlays,
            station_data,
            weather3h_data,
            thailand,
            today_error,
            weather3h_error,
        ) = create_weather_overlays()

        weather_map = folium.Map(
            location=[13.0, 101.0],
            zoom_start=5,
            min_zoom=2,
            max_zoom=18,
            control_scale=True,
            tiles=None,
        )

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            show=True,
            control=True,
        ).add_to(weather_map)

        folium.TileLayer(
            tiles="CartoDB positron",
            name="แผนที่พื้นอ่อน",
            show=False,
            control=True,
        ).add_to(weather_map)

        folium.TileLayer(
            tiles="CartoDB dark_matter",
            name="แผนที่พื้นเข้ม",
            show=False,
            control=True,
        ).add_to(weather_map)

        weather_layer_objects = {}

        for field, overlay in weather_overlays.items():
            weather_layer = ImageOverlay(
                image=overlay["image_url"],
                bounds=overlay["bounds"],
                opacity=0.72,
                name=overlay["name"],
                interactive=False,
                cross_origin=False,
                zindex=2,
                show=(field == "tmin"),
            )

            weather_layer.add_to(
                weather_map
            )

            weather_layer_objects[field] = (
                weather_layer
            )


        folium.GeoJson(
            data=thailand.to_json(),
            name="Thailand boundary",
            style_function=lambda feature: {
                "fillColor": "transparent",
                "color": "black",
                "weight": 1.3,
                "fillOpacity": 0,
            },
        ).add_to(weather_map)

        station_group = folium.FeatureGroup(
            name="Weather stations",
            show=True,
        )

        for _, station in station_data.iterrows():
            station_name = str(
                station["station"]
            ).strip()

            if not station_name:
                station_name = (
                    "Weather Station"
                )

            safe_station_name = html.escape(
                station_name
            )

            observation_time = (
                format_observation_datetime(
                    station.get(
                        "observation_datetime",
                        "",
                    )
                )
            )

            def display_value(value, unit):
                if pd.isna(value):
                    return "No data"
                return (
                    f"{float(value):.1f} {unit}"
                )

            popup_html = f"""
            <div style="
                min-width: 240px;
                font-family: Arial;
                font-size: 14px;
                line-height: 1.55;
            ">
                <b>{safe_station_name}</b><br>
                Province:
                {html.escape(str(station.get("province", "")))}<br>
                Latitude:
                {station["latitude"]:.4f}<br>
                Longitude:
                {station["longitude"]:.4f}<br>
                Temperature:
                {display_value(station.get("temperature"), "°C")}<br>
                Tmax:
                {display_value(station.get("tmax"), "°C")}<br>
                Tmin:
                {display_value(station.get("tmin"), "°C")}<br>
                Rainfall:
                {display_value(station.get("rainfall"), "mm")}<br>
                Observation time:
                {html.escape(observation_time)}
            </div>
            """


            folium.CircleMarker(
                location=[
                    station["latitude"],
                    station["longitude"],
                ],
                radius=4,
                color="white",
                weight=1,
                fill=True,
                fill_color="black",
                fill_opacity=0.9,
                pane="markerPane",
                bubbling_mouse_events=False,
                tooltip=safe_station_name,
                popup=folium.Popup(
                    popup_html,
                    max_width=300,
                ),
            ).add_to(station_group)

        station_group.add_to(weather_map)

        weather3h_group = folium.FeatureGroup(
            name="Weather3Hours stations",
            show=False,
        )

        for _, station in weather3h_data.iterrows():
            station_name = str(
                station["station"]
            ).strip()

            if not station_name:
                station_name = (
                    "Weather Station"
                )

            safe_station_name = html.escape(
                station_name
            )

            observation_time_3h = (
                format_observation_datetime(
                    station.get(
                        "observation_datetime_3h",
                        "",
                    )
                )
            )

            def display_3h_value(
                value,
                unit,
            ):
                if pd.isna(value):
                    return "No data"

                return (
                    f"{float(value):.1f} {unit}"
                )

            popup_3h_html = f"""
            <div style="
                min-width: 240px;
                font-family: Arial;
                font-size: 14px;
                line-height: 1.55;
            ">
                <b>{safe_station_name}</b><br>
                Data source: Weather3Hours<br>
                Province:
                {html.escape(str(station.get("province", "")))}<br>
                Latitude:
                {station["latitude"]:.4f}<br>
                Longitude:
                {station["longitude"]:.4f}<br>
                Air Temperature:
                {display_3h_value(
                    station.get("air_temperature_3h"),
                    "°C"
                )}<br>
                Rainfall:
                {display_3h_value(
                    station.get("rainfall_3h"),
                    "mm"
                )}<br>
                Rainfall 24 Hour:
                {display_3h_value(
                    station.get("rainfall_24h_3h"),
                    "mm"
                )}<br>
                Observation time:
                {html.escape(observation_time_3h)}
            </div>
            """

            folium.CircleMarker(
                location=[
                    station["latitude"],
                    station["longitude"],
                ],
                radius=4,
                color="#ffffff",
                weight=1,
                fill=True,
                fill_color="#1464a5",
                fill_opacity=0.95,
                pane="markerPane",
                bubbling_mouse_events=False,
                tooltip=(
                    safe_station_name
                    + " (Weather3Hours)"
                ),
                popup=folium.Popup(
                    popup_3h_html,
                    max_width=320,
                ),
            ).add_to(
                weather3h_group
            )

        weather3h_group.add_to(
            weather_map
        )

        folium.LayerControl(
            collapsed=False
        ).add_to(weather_map)

        (
            minimum_lon,
            minimum_lat,
            maximum_lon,
            maximum_lat,
        ) = thailand.total_bounds

        weather_map.fit_bounds(
            [
                [minimum_lat, minimum_lon],
                [maximum_lat, maximum_lon],
            ]
        )

        # -------------------------------------------------
        # ข้อมูลสำหรับ JavaScript
        # -------------------------------------------------

        today_record_columns = [
            "latitude",
            "longitude",
            "temperature",
            "tmax",
            "tmin",
            "rainfall",
        ]

        today_records = (
            station_data[
                today_record_columns
            ]
            .where(
                pd.notna(
                    station_data[
                        today_record_columns
                    ]
                ),
                None,
            )
            .to_dict(
                orient="records"
            )
        )

        weather3h_record_columns = [
            "latitude",
            "longitude",
            "air_temperature_3h",
            "rainfall_3h",
            "rainfall_24h_3h",
        ]

        weather3h_records = (
            weather3h_data[
                weather3h_record_columns
            ]
            .where(
                pd.notna(
                    weather3h_data[
                        weather3h_record_columns
                    ]
                ),
                None,
            )
            .to_dict(
                orient="records"
            )
        )

        station_records = (
            today_records
            + weather3h_records
        )

        stations_json = json.dumps(
            station_records,
            ensure_ascii=False,
            allow_nan=False,
        )

        thailand_geojson_object = json.loads(
            thailand.to_json()
        )

        thailand_geojson_json = json.dumps(
            thailand_geojson_object,
            ensure_ascii=False,
            allow_nan=False,
        )

        map_variable_name = (
            weather_map.get_name()
        )

        weather_layer_js_map = (
            "{"
            + ",".join(
                json.dumps(field)
                + ":"
                + layer.get_name()
                for field, layer
                in weather_layer_objects.items()
            )
            + "}"
        )

        station_layer_js_map = (
            "{"
            + json.dumps("weather_today")
            + ":"
            + station_group.get_name()
            + ","
            + json.dumps("weather_3hours")
            + ":"
            + weather3h_group.get_name()
            + "}"
        )

        layer_metadata = {
            field: {
                "name": overlay["name"],
                "label": (
                    "Interpolated "
                    + overlay["short_name"]
                ),
                "short_name": (
                    overlay["short_name"]
                ),
                "source_name": (
                    overlay["source_name"]
                ),
                "unit": overlay["unit"],
                "minimum": overlay["minimum"],
                "maximum": overlay["maximum"],
            }
            for field, overlay
            in weather_overlays.items()
        }

        layer_metadata_json = json.dumps(
            layer_metadata,
            ensure_ascii=False,
            allow_nan=False,
        )

        # -------------------------------------------------
        # ชื่อแผนที่และเวลา
        # -------------------------------------------------

        today_times = (
            station_data[
                "observation_datetime"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        today_times = today_times[
            today_times != ""
        ]

        weather3h_times = (
            weather3h_data[
                "observation_datetime_3h"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        weather3h_times = weather3h_times[
            weather3h_times != ""
        ]

        today_observation_time = (
            format_observation_datetime(
                today_times.max()
            )
            if not today_times.empty
            else "Unknown"
        )

        weather3h_observation_time = (
            format_observation_datetime(
                weather3h_times.max()
            )
            if not weather3h_times.empty
            else "Unknown"
        )

        api_warning_parts = []

        if today_error:
            api_warning_parts.append(
                "WeatherToday temporarily unavailable"
            )

        if weather3h_error:
            api_warning_parts.append(
                "Weather3Hours temporarily unavailable"
            )

        api_warning_text = " | ".join(
            api_warning_parts
        )

        warning_html = (
            f"""
            <div class="api-warning">
                {html.escape(api_warning_text)}
            </div>
            """
            if api_warning_text
            else ""
        )

        title_html = f"""
        <div class="map-title">
            Thailand Weather Interpolation Map

            <div class="map-observation-time">
                WeatherToday:
                {html.escape(today_observation_time)}
                &nbsp;|&nbsp;
                Weather3Hours:
                {html.escape(weather3h_observation_time)}
            </div>

            {warning_html}
        </div>
        """

        # -------------------------------------------------
        # ช่องค้นหา
        # -------------------------------------------------

        search_html = """
        <div id="place-search-panel">
            <div class="search-row">
                <input
                    id="place-search-input"
                    type="text"
                    maxlength="150"
                    placeholder="ค้นหาจังหวัด อำเภอ ตำบล หรือสถานที่..."
                    autocomplete="off"
                >

                <button
                    id="place-search-button"
                    type="button"
                >
                    ค้นหา
                </button>
            </div>

            <div id="place-search-status"></div>

            <div id="place-search-results"></div>

            <div class="search-attribution">
                Search data ©
                <a
                    href="https://www.openstreetmap.org/copyright"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    OpenStreetMap contributors
                </a>
            </div>
        </div>
        """

        # -------------------------------------------------
        # Legend
        # -------------------------------------------------

        legend_html = """
        <div id="temperature-legend">
            <b id="legend-title">
                WeatherToday - Tmin
            </b>

            <div
                id="legend-gradient"
                class="temperature-gradient"
            ></div>

            <div class="temperature-range">
                <span id="legend-min"></span>
                <span id="legend-max"></span>
            </div>
        </div>
        """

        # -------------------------------------------------
        # กล่องค่าเมาส์
        # -------------------------------------------------

        mouse_value_html = """
        <div id="mouse-tmin-box">
            <b>Interpolated Tmin</b><br>
            เลื่อนเมาส์ภายในประเทศไทย
        </div>

        <div id="cursor-tmin-box"></div>
        """

        export_button_html = """
        <button id="export-map-button" type="button">
            Export Publication Map
        </button>
        """

        # -------------------------------------------------
        # CSS
        # -------------------------------------------------

        custom_style = """
        <style>
            .map-title {
                position: fixed;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 9999;
                padding: 9px 18px;
                background: rgba(255,255,255,0.94);
                border-radius: 7px;
                font-family: Arial, sans-serif;
                font-weight: bold;
                font-size: 17px;
                box-shadow:
                    0 1px 6px rgba(0,0,0,0.35);
                white-space: nowrap;
            }

            .map-observation-time {
                margin-top: 3px;
                font-size: 11px;
                font-weight: normal;
            }

            .api-warning {
                margin-top: 4px;
                color: #a33;
                font-size: 11px;
                font-weight: normal;
            }

            #place-search-panel {
                position: fixed;
                top: 85px;
                left: 50px;
                z-index: 10000;
                width: min(390px, calc(100vw - 80px));
                padding: 10px;
                background: rgba(255,255,255,0.97);
                border: 1px solid #888;
                border-radius: 7px;
                font-family: Arial, sans-serif;
                box-shadow:
                    0 2px 8px rgba(0,0,0,0.30);
            }

            .search-row {
                display: flex;
                gap: 7px;
            }

            #place-search-input {
                flex: 1;
                min-width: 0;
                padding: 9px 10px;
                border: 1px solid #aaa;
                border-radius: 5px;
                font-size: 14px;
            }

            #place-search-button {
                padding: 9px 13px;
                border: 0;
                border-radius: 5px;
                background: #1769aa;
                color: white;
                font-size: 14px;
                cursor: pointer;
            }

            #place-search-button:hover {
                background: #0e4f83;
            }

            #place-search-button:disabled {
                background: #888;
                cursor: wait;
            }

            #place-search-status {
                display: none;
                margin-top: 8px;
                padding: 6px;
                border-radius: 4px;
                background: #f3f3f3;
                font-size: 13px;
            }

            #place-search-results {
                display: none;
                max-height: 270px;
                margin-top: 7px;
                overflow-y: auto;
                border: 1px solid #ccc;
                border-radius: 5px;
                background: white;
            }

            .place-result {
                padding: 9px;
                border-bottom: 1px solid #e4e4e4;
                cursor: pointer;
                font-size: 13px;
                line-height: 1.4;
            }

            .place-result:last-child {
                border-bottom: 0;
            }

            .place-result:hover {
                background: #e9f3ff;
            }

            .search-attribution {
                margin-top: 7px;
                font-size: 10px;
                color: #666;
            }

            .search-attribution a {
                color: #555;
            }

            #temperature-legend {
                position: fixed;
                bottom: 35px;
                left: 35px;
                z-index: 9999;
                width: 220px;
                padding: 12px;
                background: rgba(255,255,255,0.95);
                border: 2px solid #555;
                border-radius: 7px;
                font-family: Arial, sans-serif;
                font-size: 13px;
                box-shadow:
                    0 1px 7px rgba(0,0,0,0.35);
            }

            .temperature-gradient {
                width: 200px;
                height: 16px;
                margin-top: 8px;
                margin-bottom: 5px;
                background: linear-gradient(
                    to right,
                    #30123b,
                    #4145ab,
                    #4675ed,
                    #39a2fc,
                    #1bcfd4,
                    #24eca6,
                    #61fc6c,
                    #a4fc3c,
                    #d1e834,
                    #f9ba38,
                    #f66b19,
                    #d93806,
                    #7a0403
                );
            }

            .temperature-range {
                display: flex;
                justify-content: space-between;
            }

            #mouse-tmin-box {
                position: fixed;
                right: 35px;
                bottom: 35px;
                z-index: 9999;
                min-width: 220px;
                padding: 12px;
                background: rgba(255,255,255,0.96);
                border: 2px solid #555;
                border-radius: 7px;
                font-family: Arial, sans-serif;
                font-size: 14px;
                line-height: 1.5;
                box-shadow:
                    0 1px 7px rgba(0,0,0,0.35);
            }

            #export-map-button {
                position: fixed;
                top: 285px;
                right: 20px;
                z-index: 10050;
                padding: 11px 15px;
                border: 2px solid white;
                border-radius: 6px;
                background: #176b45;
                color: white;
                font-family: Arial, sans-serif;
                font-size: 14px;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 2px 7px rgba(0,0,0,0.30);
            }

            #export-map-button:hover {
                background: #145c39;
            }

            #export-map-button:disabled {
                background: #777;
                cursor: wait;
            }

            body.exporting-map #place-search-panel,
            body.exporting-map .leaflet-control-layers,
            body.exporting-map .leaflet-control-zoom,
            body.exporting-map #export-map-button,
            body.exporting-map #cursor-tmin-box {
                display: none !important;
            }

            #cursor-tmin-box {
                display: none;
                position: fixed;
                z-index: 10001;
                pointer-events: none;
                padding: 6px 9px;
                background: rgba(0,0,0,0.82);
                color: white;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 13px;
                white-space: nowrap;
            }

            @media (max-width: 700px) {
                .map-title {
                    max-width: calc(100vw - 40px);
                    overflow: hidden;
                    text-overflow: ellipsis;
                    font-size: 14px;
                }

                #place-search-panel {
                    top: 65px;
                    left: 10px;
                    width: calc(100vw - 40px);
                }

                #export-map-button {
                    top: 125px;
                    right: 10px;
                    padding: 8px 10px;
                    font-size: 12px;
                }

                #temperature-legend {
                    left: 10px;
                    bottom: 10px;
                    width: 175px;
                }

                .temperature-gradient {
                    width: 155px;
                }

                #mouse-tmin-box {
                    right: 10px;
                    bottom: 10px;
                    min-width: 155px;
                    font-size: 12px;
                }
            }
        </style>
        """

        # -------------------------------------------------
        # JavaScript
        # -------------------------------------------------

        custom_script = f"""
        <script>
        (function() {{

            const stations = {stations_json};

            const layerMetadata =
                {layer_metadata_json};

            let currentField = "tmin";

            const weatherLayers =
                {weather_layer_js_map};

            const stationLayers =
                {station_layer_js_map};

            let changingExclusiveLayer = false;

            const thailandGeoJSON =
                {thailand_geojson_json};

            const mapObject =
                {map_variable_name};

            const valueBox =
                document.getElementById(
                    "mouse-tmin-box"
                );

            const cursorBox =
                document.getElementById(
                    "cursor-tmin-box"
                );

            const searchInput =
                document.getElementById(
                    "place-search-input"
                );

            const searchButton =
                document.getElementById(
                    "place-search-button"
                );

            const searchStatus =
                document.getElementById(
                    "place-search-status"
                );

            const searchResults =
                document.getElementById(
                    "place-search-results"
                );

            const exportMapButton =
                document.getElementById(
                    "export-map-button"
                )
                ||
                (() => {{
                    const button =
                        document.createElement(
                            "button"
                        );

                    button.id =
                        "export-map-button";

                    button.type =
                        "button";

                    button.textContent =
                        "Export Publication Map";

                    document.body.appendChild(
                        button
                    );

                    return button;
                }})();

            let searchMarker = null;


            async function exportPublicationMap() {{
                const exportUrl =
                    "/export/publication?layer="
                    + encodeURIComponent(
                        currentField
                    );

                exportMapButton.disabled = true;
                exportMapButton.textContent =
                    "Creating map...";

                try {{
                    const response = await fetch(
                        exportUrl,
                        {{
                            method: "GET"
                        }}
                    );

                    if (!response.ok) {{
                        let message =
                            "ไม่สามารถ Export แผนที่ได้";

                        try {{
                            const errorData =
                                await response.json();

                            if (
                                errorData
                                &&
                                errorData.detail
                            ) {{
                                message =
                                    errorData.detail;
                            }}
                        }}
                        catch (parseError) {{
                            // ใช้ข้อความเริ่มต้น
                        }}

                        throw new Error(message);
                    }}

                    const imageBlob =
                        await response.blob();

                    const disposition =
                        response.headers.get(
                            "content-disposition"
                        )
                        || "";

                    const filenameMatch =
                        disposition.match(
                            /filename="?([^"]+)"?/i
                        );

                    const filename =
                        filenameMatch
                        ? filenameMatch[1]
                        : (
                            "Thailand_Weather_Map_"
                            + currentField
                            + ".png"
                        );

                    const objectUrl =
                        URL.createObjectURL(
                            imageBlob
                        );

                    const downloadLink =
                        document.createElement(
                            "a"
                        );

                    downloadLink.href =
                        objectUrl;

                    downloadLink.download =
                        filename;

                    document.body.appendChild(
                        downloadLink
                    );

                    downloadLink.click();
                    downloadLink.remove();

                    URL.revokeObjectURL(
                        objectUrl
                    );
                }}
                catch (error) {{
                    console.error(error);

                    alert(
                        error.message
                        ||
                        "Export แผนที่ไม่สำเร็จ"
                    );
                }}
                finally {{
                    exportMapButton.disabled =
                        false;

                    exportMapButton.textContent =
                        "Export Publication Map";
                }}
            }}


            exportMapButton.addEventListener(
                "click",
                exportPublicationMap
            );


            // =============================================
            // ป้องกันข้อความจาก API ถูกตีความเป็น HTML
            // =============================================

            function escapeHtml(text) {{
                const element =
                    document.createElement("div");

                element.textContent =
                    String(text);

                return element.innerHTML;
            }}


            // =============================================
            // แสดงสถานะการค้นหา
            // =============================================

            function showSearchStatus(
                message,
                isError
            ) {{
                searchStatus.style.display =
                    "block";

                searchStatus.textContent =
                    message;

                searchStatus.style.color =
                    isError ? "#9c1c1c" : "#333";

                searchStatus.style.background =
                    isError ? "#ffeaea" : "#f3f3f3";
            }}


            function getCurrentLayer() {{
                return (
                    layerMetadata[currentField]
                    || layerMetadata.tmin
                );
            }}


            function getCurrentLabel() {{
                return getCurrentLayer().label;
            }}


            function getCurrentShortName() {{
                return getCurrentLayer().short_name;
            }}


            function getCurrentUnit() {{
                return getCurrentLayer().unit;
            }}


            function setCurrentFieldFromLayerName(
                layerName
            ) {{
                for (
                    const [field, metadata]
                    of Object.entries(layerMetadata)
                ) {{
                    if (metadata.name === layerName) {{
                        currentField = field;
                        return true;
                    }}
                }}

                return false;
            }}


            function updateLegend() {{
                const metadata =
                    getCurrentLayer();

                document.getElementById(
                    "legend-title"
                ).textContent =
                    metadata.name;

                const decimals =
                    metadata.unit === "mm"
                    ? 1
                    : 0;

                document.getElementById(
                    "legend-min"
                ).textContent =
                    Number(metadata.minimum)
                    .toFixed(decimals)
                    + " "
                    + metadata.unit;

                document.getElementById(
                    "legend-max"
                ).textContent =
                    Number(metadata.maximum)
                    .toFixed(decimals)
                    + " "
                    + metadata.unit;

                const gradient =
                    document.getElementById(
                        "legend-gradient"
                    );

                if (metadata.unit === "mm") {{
                    gradient.style.background =
                        "linear-gradient(to right,"
                        + "#f7fbff,#deebf7,#c6dbef,"
                        + "#9ecae1,#6baed6,#4292c6,"
                        + "#2171b5,#08519c,#08306b)";
                }}
                else {{
                    gradient.style.background =
                        "linear-gradient(to right,"
                        + "#30123b,#4145ab,#4675ed,"
                        + "#39a2fc,#1bcfd4,#24eca6,"
                        + "#61fc6c,#a4fc3c,#d1e834,"
                        + "#f9ba38,#f66b19,#d93806,"
                        + "#7a0403)";
                }}
            }}


            // =============================================
            // ตรวจจุดใน Polygon
            // =============================================

            function pointInRing(
                longitude,
                latitude,
                ring
            ) {{
                let inside = false;

                for (
                    let index = 0,
                        previousIndex =
                            ring.length - 1;
                    index < ring.length;
                    previousIndex = index++
                ) {{
                    const currentLongitude =
                        ring[index][0];

                    const currentLatitude =
                        ring[index][1];

                    const previousLongitude =
                        ring[previousIndex][0];

                    const previousLatitude =
                        ring[previousIndex][1];

                    const crosses =
                        (
                            (
                                currentLatitude >
                                latitude
                            ) !==
                            (
                                previousLatitude >
                                latitude
                            )
                        )
                        &&
                        (
                            longitude <
                            (
                                (
                                    previousLongitude -
                                    currentLongitude
                                )
                                *
                                (
                                    latitude -
                                    currentLatitude
                                )
                            )
                            /
                            (
                                (
                                    previousLatitude -
                                    currentLatitude
                                )
                                || 1e-12
                            )
                            +
                            currentLongitude
                        );

                    if (crosses) {{
                        inside = !inside;
                    }}
                }}

                return inside;
            }}


            function pointInPolygon(
                longitude,
                latitude,
                polygonCoordinates
            ) {{
                if (
                    !polygonCoordinates ||
                    polygonCoordinates.length === 0
                ) {{
                    return false;
                }}

                if (
                    !pointInRing(
                        longitude,
                        latitude,
                        polygonCoordinates[0]
                    )
                ) {{
                    return false;
                }}

                for (
                    let holeIndex = 1;
                    holeIndex <
                    polygonCoordinates.length;
                    holeIndex++
                ) {{
                    if (
                        pointInRing(
                            longitude,
                            latitude,
                            polygonCoordinates[
                                holeIndex
                            ]
                        )
                    ) {{
                        return false;
                    }}
                }}

                return true;
            }}


            function pointInsideThailand(
                latitude,
                longitude
            ) {{
                const features =
                    thailandGeoJSON.features || [];

                for (const feature of features) {{
                    if (
                        !feature.geometry ||
                        !feature.geometry.coordinates
                    ) {{
                        continue;
                    }}

                    const geometry =
                        feature.geometry;

                    if (
                        geometry.type ===
                        "Polygon"
                    ) {{
                        if (
                            pointInPolygon(
                                longitude,
                                latitude,
                                geometry.coordinates
                            )
                        ) {{
                            return true;
                        }}
                    }}

                    if (
                        geometry.type ===
                        "MultiPolygon"
                    ) {{
                        for (
                            const polygonCoordinates
                            of geometry.coordinates
                        ) {{
                            if (
                                pointInPolygon(
                                    longitude,
                                    latitude,
                                    polygonCoordinates
                                )
                            ) {{
                                return true;
                            }}
                        }}
                    }}
                }}

                return false;
            }}


            // =============================================
            // IDW ที่ตำแหน่งเมาส์หรือผลค้นหา
            // =============================================

            function calculateIDW(
                latitude,
                longitude
            ) {{
                const power = 2.0;
                const nearestCount = 8;

                const distances =
                    stations.map(
                        function(station) {{
                            const dx =
                                longitude -
                                station.longitude;

                            const dy =
                                latitude -
                                station.latitude;

                            return {{
                                distance: Math.sqrt(
                                    dx * dx +
                                    dy * dy
                                ),
                                value:
                                    (
                                        station[currentField]
                                        === null
                                        ||
                                        station[currentField]
                                        === undefined
                                    )
                                    ? NaN
                                    : Number(
                                        station[currentField]
                                    )
                            }};
                        }}
                    );

                distances.sort(
                    function(first, second) {{
                        return (
                            first.distance -
                            second.distance
                        );
                    }}
                );

                const validDistances =
                    distances.filter(
                        function(item) {{
                            return Number.isFinite(
                                item.value
                            );
                        }}
                    );

                const nearestStations =
                    validDistances.slice(
                        0,
                        Math.min(
                            nearestCount,
                            validDistances.length
                        )
                    );

                if (
                    nearestStations.length === 0
                ) {{
                    return null;
                }}

                if (
                    nearestStations[0].distance <
                    0.000001
                ) {{
                    return (
                        nearestStations[0].value
                    );
                }}

                let weightedValueSum = 0;
                let weightSum = 0;

                nearestStations.forEach(
                    function(item) {{
                        const safeDistance =
                            Math.max(
                                item.distance,
                                0.000001
                            );

                        const weight =
                            1 / Math.pow(
                                safeDistance,
                                power
                            );

                        weightedValueSum +=
                            weight * item.value;

                        weightSum += weight;
                    }}
                );

                if (weightSum === 0) {{
                    return null;
                }}

                return (
                    weightedValueSum /
                    weightSum
                );
            }}


            // =============================================
            // เลือกผลการค้นหา
            // =============================================

            function selectSearchResult(
                result
            ) {{
                const latitude =
                    Number(result.latitude);

                const longitude =
                    Number(result.longitude);

                if (
                    !Number.isFinite(latitude) ||
                    !Number.isFinite(longitude)
                ) {{
                    showSearchStatus(
                        "พิกัดของสถานที่ไม่ถูกต้อง",
                        true
                    );

                    return;
                }}

                searchResults.style.display =
                    "none";

                searchInput.value =
                    result.display_name;

                if (searchMarker !== null) {{
                    mapObject.removeLayer(
                        searchMarker
                    );
                }}

                const insideThailand =
                    pointInsideThailand(
                        latitude,
                        longitude
                    );

                const interpolatedTmin =
                    insideThailand
                    ? calculateIDW(
                        latitude,
                        longitude
                    )
                    : null;

                let popupHtml =
                    "<div style='" +
                    "min-width:220px;" +
                    "font-family:Arial;" +
                    "line-height:1.5;" +
                    "'>" +
                    "<b>" +
                    escapeHtml(
                        result.display_name
                    ) +
                    "</b><br>" +
                    "Lat: " +
                    latitude.toFixed(5) +
                    "<br>" +
                    "Lon: " +
                    longitude.toFixed(5);

                if (
                    interpolatedTmin !== null &&
                    Number.isFinite(
                        interpolatedTmin
                    )
                ) {{
                    popupHtml +=
                        "<br><b>" +
                        getCurrentLabel() +
                        ": " +
                        interpolatedTmin.toFixed(1) +
                        " " +
                        getCurrentUnit() +
                        "</b>";
                }}

                popupHtml += "</div>";

                searchMarker = L.marker(
                    [latitude, longitude]
                )
                .addTo(mapObject)
                .bindPopup(popupHtml)
                .openPopup();

                const boundingBox =
                    result.boundingbox;

                if (
                    Array.isArray(boundingBox) &&
                    boundingBox.length === 4
                ) {{
                    const south =
                        Number(boundingBox[0]);

                    const north =
                        Number(boundingBox[1]);

                    const west =
                        Number(boundingBox[2]);

                    const east =
                        Number(boundingBox[3]);

                    if (
                        Number.isFinite(south) &&
                        Number.isFinite(north) &&
                        Number.isFinite(west) &&
                        Number.isFinite(east) &&
                        north > south &&
                        east > west
                    ) {{
                        mapObject.fitBounds(
                            [
                                [south, west],
                                [north, east],
                            ],
                            {{
                                maxZoom: 14,
                                padding: [40, 40]
                            }}
                        );
                    }}
                    else {{
                        mapObject.setView(
                            [latitude, longitude],
                            13
                        );
                    }}
                }}
                else {{
                    mapObject.setView(
                        [latitude, longitude],
                        13
                    );
                }}

                if (
                    insideThailand &&
                    interpolatedTmin !== null
                ) {{
                    valueBox.innerHTML =
                        "<b>Search location</b><br>" +
                        getCurrentShortName() +
                        ": " +
                        interpolatedTmin.toFixed(1) +
                        " " +
                        getCurrentUnit() +
                        "<br>" +
                        "Lat: " +
                        latitude.toFixed(4) +
                        "<br>" +
                        "Lon: " +
                        longitude.toFixed(4);
                }}
            }}


            // =============================================
            // แสดงรายการผลการค้นหา
            // =============================================

            function renderSearchResults(
                results
            ) {{
                searchResults.innerHTML = "";

                if (
                    !Array.isArray(results) ||
                    results.length === 0
                ) {{
                    searchResults.style.display =
                        "none";

                    showSearchStatus(
                        "ไม่พบสถานที่ที่ค้นหา",
                        true
                    );

                    return;
                }}

                results.forEach(
                    function(result) {{
                        const resultElement =
                            document.createElement(
                                "div"
                            );

                        resultElement.className =
                            "place-result";

                        resultElement.textContent =
                            result.display_name;

                        resultElement.addEventListener(
                            "click",
                            function() {{
                                selectSearchResult(
                                    result
                                );
                            }}
                        );

                        searchResults.appendChild(
                            resultElement
                        );
                    }}
                );

                searchResults.style.display =
                    "block";

                showSearchStatus(
                    "พบ " +
                    results.length +
                    " รายการ กรุณาเลือกรายการ",
                    false
                );
            }}


            // =============================================
            // ส่งคำค้นหาไป FastAPI
            // =============================================

            async function searchPlace() {{
                const query =
                    searchInput.value.trim();

                if (query.length < 2) {{
                    showSearchStatus(
                        "กรุณาพิมพ์อย่างน้อย 2 ตัวอักษร",
                        true
                    );

                    return;
                }}

                searchButton.disabled = true;

                searchResults.style.display =
                    "none";

                showSearchStatus(
                    "กำลังค้นหา...",
                    false
                );

                try {{
                    const response = await fetch(
                        "/api/search?q=" +
                        encodeURIComponent(query),
                        {{
                            method: "GET",
                            headers: {{
                                "Accept":
                                    "application/json"
                            }}
                        }}
                    );

                    let data = null;

                    try {{
                        data = await response.json();
                    }}
                    catch (parseError) {{
                        throw new Error(
                            "เซิร์ฟเวอร์ตอบกลับไม่ถูกต้อง"
                        );
                    }}

                    if (!response.ok) {{
                        const message =
                            data &&
                            data.detail
                            ? data.detail
                            : "ค้นหาสถานที่ไม่สำเร็จ";

                        throw new Error(message);
                    }}

                    renderSearchResults(
                        data.results
                    );
                }}
                catch (error) {{
                    console.error(error);

                    showSearchStatus(
                        error.message ||
                        "ไม่สามารถค้นหาสถานที่ได้",
                        true
                    );
                }}
                finally {{
                    searchButton.disabled =
                        false;
                }}
            }}


            searchButton.addEventListener(
                "click",
                searchPlace
            );

            searchInput.addEventListener(
                "keydown",
                function(event) {{
                    if (event.key === "Enter") {{
                        event.preventDefault();

                        searchPlace();
                    }}
                }}
            );


            function findWeatherFieldByLayer(
                selectedLayer
            ) {{
                for (
                    const [field, layer]
                    of Object.entries(
                        weatherLayers
                    )
                ) {{
                    if (layer === selectedLayer) {{
                        return field;
                    }}
                }}

                return null;
            }}


            function findStationKeyByLayer(
                selectedLayer
            ) {{
                for (
                    const [key, layer]
                    of Object.entries(
                        stationLayers
                    )
                ) {{
                    if (layer === selectedLayer) {{
                        return key;
                    }}
                }}

                return null;
            }}


            function initializeExclusiveRadioControls() {{
                const controlContainer =
                    document.querySelector(
                        ".leaflet-control-layers"
                    );

                if (!controlContainer) {{
                    return;
                }}

                const labels =
                    controlContainer.querySelectorAll(
                        "label"
                    );

                labels.forEach(
                    function(label) {{
                        const input =
                            label.querySelector(
                                "input"
                            );

                        const labelText =
                            label.textContent.trim();

                        if (!input) {{
                            return;
                        }}

                        let weatherField = null;

                        for (
                            const [field, metadata]
                            of Object.entries(
                                layerMetadata
                            )
                        ) {{
                            if (
                                metadata.name ===
                                labelText
                            ) {{
                                weatherField = field;
                                break;
                            }}
                        }}

                        if (weatherField !== null) {{
                            input.type = "radio";
                            input.name =
                                "weather-data-layer";

                            input.addEventListener(
                                "change",
                                function() {{
                                    if (!input.checked) {{
                                        return;
                                    }}

                                    changingExclusiveLayer =
                                        true;

                                    try {{
                                        for (
                                            const [field, layer]
                                            of Object.entries(
                                                weatherLayers
                                            )
                                        ) {{
                                            if (
                                                field ===
                                                weatherField
                                            ) {{
                                                if (
                                                    !mapObject.hasLayer(
                                                        layer
                                                    )
                                                ) {{
                                                    mapObject.addLayer(
                                                        layer
                                                    );
                                                }}
                                            }}
                                            else if (
                                                mapObject.hasLayer(
                                                    layer
                                                )
                                            ) {{
                                                mapObject.removeLayer(
                                                    layer
                                                );
                                            }}
                                        }}

                                        currentField =
                                            weatherField;

                                        valueBox.innerHTML =
                                            "<b>" +
                                            getCurrentLabel() +
                                            "</b><br>" +
                                            "เลื่อนเมาส์ภายในประเทศไทย";

                                        updateLegend();
                                    }}
                                    finally {{
                                        changingExclusiveLayer =
                                            false;
                                    }}
                                }}
                            );

                            return;
                        }}

                        if (
                            labelText ===
                            "Weather stations"
                            ||
                            labelText ===
                            "Weather3Hours stations"
                        ) {{
                            input.type = "radio";
                            input.name =
                                "weather-station-layer";

                            input.addEventListener(
                                "change",
                                function() {{
                                    if (!input.checked) {{
                                        return;
                                    }}

                                    const selectedKey =
                                        (
                                            labelText ===
                                            "Weather stations"
                                        )
                                        ? "weather_today"
                                        : "weather_3hours";

                                    changingExclusiveLayer =
                                        true;

                                    try {{
                                        for (
                                            const [key, layer]
                                            of Object.entries(
                                                stationLayers
                                            )
                                        ) {{
                                            if (
                                                key ===
                                                selectedKey
                                            ) {{
                                                if (
                                                    !mapObject.hasLayer(
                                                        layer
                                                    )
                                                ) {{
                                                    mapObject.addLayer(
                                                        layer
                                                    );
                                                }}
                                            }}
                                            else if (
                                                mapObject.hasLayer(
                                                    layer
                                                )
                                            ) {{
                                                mapObject.removeLayer(
                                                    layer
                                                );
                                            }}
                                        }}
                                    }}
                                    finally {{
                                        changingExclusiveLayer =
                                            false;
                                    }}
                                }}
                            );
                        }}
                    }}
                );
            }}


            initializeExclusiveRadioControls();


            mapObject.on(
                "overlayadd",
                function(event) {{
                    if (changingExclusiveLayer) {{
                        return;
                    }}

                    changingExclusiveLayer = true;

                    try {{
                        const selectedWeatherField =
                            findWeatherFieldByLayer(
                                event.layer
                            );

                        if (
                            selectedWeatherField !== null
                        ) {{
                            for (
                                const [field, layer]
                                of Object.entries(
                                    weatherLayers
                                )
                            ) {{
                                if (
                                    field !==
                                    selectedWeatherField
                                    &&
                                    mapObject.hasLayer(
                                        layer
                                    )
                                ) {{
                                    mapObject.removeLayer(
                                        layer
                                    );
                                }}
                            }}

                            currentField =
                                selectedWeatherField;

                            valueBox.innerHTML =
                                "<b>" +
                                getCurrentLabel() +
                                "</b><br>" +
                                "เลื่อนเมาส์ภายในประเทศไทย";

                            updateLegend();
                        }}

                        const selectedStationKey =
                            findStationKeyByLayer(
                                event.layer
                            );

                        if (
                            selectedStationKey !== null
                        ) {{
                            for (
                                const [key, layer]
                                of Object.entries(
                                    stationLayers
                                )
                            ) {{
                                if (
                                    key !==
                                    selectedStationKey
                                    &&
                                    mapObject.hasLayer(
                                        layer
                                    )
                                ) {{
                                    mapObject.removeLayer(
                                        layer
                                    );
                                }}
                            }}
                        }}
                    }}
                    finally {{
                        changingExclusiveLayer = false;
                    }}
                }}
            );

            function syncExclusiveRadioControls() {{
                const controlContainer =
                    document.querySelector(
                        ".leaflet-control-layers"
                    );

                if (!controlContainer) {{
                    return;
                }}

                controlContainer
                .querySelectorAll("label")
                .forEach(
                    function(label) {{
                        const input =
                            label.querySelector(
                                "input"
                            );

                        const labelText =
                            label.textContent.trim();

                        if (!input) {{
                            return;
                        }}

                        for (
                            const [field, metadata]
                            of Object.entries(
                                layerMetadata
                            )
                        ) {{
                            if (
                                metadata.name ===
                                labelText
                            ) {{
                                input.checked =
                                    mapObject.hasLayer(
                                        weatherLayers[field]
                                    );
                                return;
                            }}
                        }}

                        if (
                            labelText ===
                            "Weather stations"
                        ) {{
                            input.checked =
                                mapObject.hasLayer(
                                    stationLayers[
                                        "weather_today"
                                    ]
                                );
                        }}

                        if (
                            labelText ===
                            "Weather3Hours stations"
                        ) {{
                            input.checked =
                                mapObject.hasLayer(
                                    stationLayers[
                                        "weather_3hours"
                                    ]
                                );
                        }}
                    }}
                );
            }}


            mapObject.on(
                "overlayadd",
                syncExclusiveRadioControls
            );

            mapObject.on(
                "overlayremove",
                syncExclusiveRadioControls
            );

            syncExclusiveRadioControls();
            updateLegend();


            // =============================================
            // ค่าตาม Layer ที่เลือก
            // =============================================

            mapObject.on(
                "mousemove",
                function(event) {{
                    const latitude =
                        event.latlng.lat;

                    const longitude =
                        event.latlng.lng;

                    const originalEvent =
                        event.originalEvent;

                    const insideThailand =
                        pointInsideThailand(
                            latitude,
                            longitude
                        );

                    if (!insideThailand) {{
                        valueBox.innerHTML =
                            "<b>" +
                            getCurrentLabel() +
                            "</b><br>" +
                            "อยู่นอกพื้นที่ประเทศไทย";

                        cursorBox.style.display =
                            "none";

                        return;
                    }}

                    const interpolatedValue =
                        calculateIDW(
                            latitude,
                            longitude
                        );

                    if (
                        interpolatedValue === null ||
                        !Number.isFinite(
                            interpolatedValue
                        )
                    ) {{
                        valueBox.innerHTML =
                            "<b>" +
                            getCurrentLabel() +
                            "</b><br>" +
                            "ไม่สามารถคำนวณค่าได้";

                        cursorBox.style.display =
                            "none";

                        return;
                    }}

                    valueBox.innerHTML =
                        "<b>" +
                        getCurrentLabel() +
                        "</b><br>" +
                        getCurrentShortName() +
                        ": " +
                        interpolatedValue.toFixed(1) +
                        " " +
                        getCurrentUnit() +
                        "<br>" +
                        "Lat: " +
                        latitude.toFixed(4) +
                        "<br>" +
                        "Lon: " +
                        longitude.toFixed(4);

                    cursorBox.textContent =
                        getCurrentShortName() +
                        ": " +
                        interpolatedValue.toFixed(1) +
                        " " +
                        getCurrentUnit();

                    cursorBox.style.display =
                        "block";

                    if (originalEvent) {{
                        cursorBox.style.left =
                            (
                                originalEvent.clientX +
                                14
                            ) +
                            "px";

                        cursorBox.style.top =
                            (
                                originalEvent.clientY +
                                14
                            ) +
                            "px";
                    }}
                }}
            );


            mapObject.on(
                "mouseout",
                function() {{
                    cursorBox.style.display =
                        "none";
                }}
            );

        }})();
        </script>
        """

        # -------------------------------------------------
        # Render HTML
        # -------------------------------------------------

        map_html = (
            weather_map.get_root().render()
        )

        map_html = map_html.replace(
            "</head>",
            custom_style
            + """"""
            + "</head>",
        )

        map_html = map_html.replace(
            "</body>",
            title_html
            + search_html
            + legend_html
            + mouse_value_html
            + export_button_html
            + "</body>",
        )

        # วาง JavaScript หลังสคริปต์ Folium
        map_html = map_html.replace(
            "</html>",
            custom_script
            + "</html>",
        )

        return HTMLResponse(
            content=map_html
        )

    except Exception as error:
        safe_error = html.escape(
            str(error)
        )

        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="th">
            <head>
                <meta charset="UTF-8">
                <title>เกิดข้อผิดพลาด</title>
            </head>

            <body style="
                padding: 30px;
                background: #eeeeee;
                font-family: Arial, sans-serif;
            ">
                <div style="
                    max-width: 950px;
                    margin: auto;
                    padding: 25px;
                    background: white;
                    border-left: 6px solid #d32f2f;
                ">
                    <h2>
                        ไม่สามารถสร้างแผนที่ได้
                    </h2>

                    <pre style="
                        white-space: pre-wrap;
                        overflow-wrap: anywhere;
                    ">{safe_error}</pre>
                </div>
            </body>
            </html>
            """,
            status_code=500,
        )


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
