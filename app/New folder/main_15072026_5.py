from __future__ import annotations

from pathlib import Path
from datetime import datetime
from threading import Lock
import base64
import html
import io
import json
import time
import xml.etree.ElementTree as ET

import folium
import geopandas as gpd
import httpx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from folium.raster_layers import ImageOverlay
from scipy.spatial import cKDTree
from shapely import intersects, points


# =========================================================
# 1. FastAPI
# =========================================================

app = FastAPI(
    title="Thailand Minimum Temperature Map",
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


# =========================================================
# 5. อ่านข้อมูลสถานีจาก TMD API
# =========================================================

def load_station_data() -> pd.DataFrame:
    """
    ดึงชื่อสถานี Latitude Longitude และ Tmin
    จาก TMD WeatherToday API ซึ่งตอบกลับเป็น XML
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
            response = client.get(
                TMD_API_URL
            )

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
        root = ET.fromstring(
            response.content
        )
    except ET.ParseError as error:
        raise RuntimeError(
            "ข้อมูลจาก TMD API ไม่ใช่ XML ที่อ่านได้"
        ) from error

    station_records: list[dict] = []

    for station in root.findall(
        ".//Station"
    ):
        station_name = (
            get_xml_text(
                station,
                "StationNameEnglish",
            )
            or get_xml_text(
                station,
                "StationNameThai",
            )
            or "Weather Station"
        )

        longitude = to_float(
            get_xml_text(
                station,
                "Longitude",
            )
        )

        latitude = to_float(
            get_xml_text(
                station,
                "Latitude",
            )
        )

        tmin = to_float(
            get_xml_text(
                station,
                "Observation/MinTemperature",
            )
        )

        if tmin is None:
            tmin = to_float(
                get_xml_text(
                    station,
                    "Observation/TemperatureMin",
                )
            )

        if tmin is None:
            tmin = to_float(
                get_xml_text(
                    station,
                    "Observation/MinimumTemperature",
                )
            )

        observation_datetime = get_xml_text(
            station,
            "Observation/DateTime",
        )

        if (
            longitude is None
            or latitude is None
            or tmin is None
        ):
            continue

        if not (
            96.0 <= longitude <= 107.0
            and 4.0 <= latitude <= 22.0
        ):
            continue

        if not (
            -10.0 <= tmin <= 50.0
        ):
            continue

        station_records.append(
            {
                "station": station_name,
                "longitude": longitude,
                "latitude": latitude,
                "tmin": tmin,
                "observation_datetime": (
                    observation_datetime or ""
                ),
            }
        )

    station_data = pd.DataFrame(
        station_records
    )

    if station_data.empty:
        raise ValueError(
            "ไม่พบข้อมูลสถานีที่มี Latitude, Longitude "
            "และ Tmin จาก TMD API"
        )

    station_data = (
        station_data
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

    return station_data



def format_observation_datetime(
    datetime_text: str,
) -> str:
    value = str(datetime_text or "").strip()

    if not value:
        return "Unknown"

    for datetime_format in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            parsed_datetime = datetime.strptime(
                value,
                datetime_format,
            )
            return parsed_datetime.strftime(
                "%d %b %Y %H:%M:%S"
            )
        except ValueError:
            continue

    return value


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

def create_temperature_overlay():
    """
    สร้างภาพ Tmin แบบโปร่งใสสำหรับ Folium
    """

    station_data = load_station_data()
    thailand = load_thailand_boundary()

    station_lon = station_data[
        "longitude"
    ].to_numpy()

    station_lat = station_data[
        "latitude"
    ].to_numpy()

    station_tmin = station_data[
        "tmin"
    ].to_numpy()

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

    grid_tmin = idw_interpolation(
        station_lon=station_lon,
        station_lat=station_lat,
        station_value=station_tmin,
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

    masked_tmin = np.ma.masked_where(
        ~thailand_mask,
        grid_tmin,
    )

    minimum_temperature = float(
        np.floor(station_tmin.min())
    )

    maximum_temperature = float(
        np.ceil(station_tmin.max())
    )

    if minimum_temperature == maximum_temperature:
        maximum_temperature += 1.0

    levels = np.linspace(
        minimum_temperature,
        maximum_temperature,
        27,
    )

    figure, axis = plt.subplots(
        figsize=(8, 12),
        dpi=110,
    )

    axis.contourf(
        grid_lon,
        grid_lat,
        masked_tmin,
        levels=levels,
        cmap="turbo",
        vmin=minimum_temperature,
        vmax=maximum_temperature,
        extend="both",
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

    plt.close(figure)

    image_buffer.seek(0)

    encoded_image = base64.b64encode(
        image_buffer.read()
    ).decode("utf-8")

    image_url = (
        "data:image/png;base64,"
        + encoded_image
    )

    image_bounds = [
        [minimum_lat, minimum_lon],
        [maximum_lat, maximum_lon],
    ]

    return (
        image_url,
        image_bounds,
        station_data,
        thailand,
        minimum_temperature,
        maximum_temperature,
    )


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
                Thailand Minimum Temperature Map
            </h1>

            <p>
                แผนที่อุณหภูมิต่ำสุดแบบโต้ตอบจาก TMD API
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
            image_url,
            image_bounds,
            station_data,
            thailand,
            minimum_temperature,
            maximum_temperature,
        ) = create_temperature_overlay()

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

        ImageOverlay(
            image=image_url,
            bounds=image_bounds,
            opacity=0.72,
            name="Interpolated Tmin",
            interactive=False,
            cross_origin=False,
            zindex=2,
        ).add_to(weather_map)

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

            observation_datetime = (
                format_observation_datetime(
                    station.get(
                        "observation_datetime",
                        "",
                    )
                )
            )

            safe_observation_datetime = html.escape(
                observation_datetime
            )

            popup_html = f"""
            <div style="
                min-width: 190px;
                font-family: Arial;
                font-size: 14px;
                line-height: 1.5;
            ">
                <b>{safe_station_name}</b><br>
                Latitude:
                {station["latitude"]:.4f}<br>
                Longitude:
                {station["longitude"]:.4f}<br>
                Observed Tmin:
                {station["tmin"]:.1f} °C<br>
                Observation Time:
                {safe_observation_datetime}
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
                tooltip=(
                    f"{safe_station_name}: "
                    f'{station["tmin"]:.1f} °C'
                ),
                popup=folium.Popup(
                    popup_html,
                    max_width=300,
                ),
            ).add_to(station_group)

        station_group.add_to(weather_map)

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

        station_records = station_data[
            [
                "latitude",
                "longitude",
                "tmin",
            ]
        ].to_dict(
            orient="records"
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

        # -------------------------------------------------
        # ชื่อแผนที่และเวลาข้อมูลล่าสุด
        # -------------------------------------------------

        observation_times = (
            station_data["observation_datetime"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        observation_times = observation_times[
            observation_times != ""
        ]

        if not observation_times.empty:
            parsed_observation_times = pd.to_datetime(
                observation_times,
                errors="coerce",
            ).dropna()

            if not parsed_observation_times.empty:
                latest_observation_text = (
                    parsed_observation_times.max().strftime(
                        "%d %b %Y %H:%M:%S"
                    )
                )
            else:
                latest_observation_text = (
                    format_observation_datetime(
                        observation_times.iloc[0]
                    )
                )
        else:
            latest_observation_text = "Unknown"

        safe_latest_observation_text = html.escape(
            latest_observation_text
        )

        title_html = f"""
        <div class="map-title">
            Interpolated Minimum Temperature in Thailand

            <div style="
                margin-top: 3px;
                font-size: 12px;
                font-weight: normal;
            ">
                Observation Time:
                {safe_latest_observation_text}
            </div>
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

        legend_html = f"""
        <div id="temperature-legend">
            <b>Minimum Temperature</b>

            <div class="temperature-gradient"></div>

            <div class="temperature-range">
                <span>
                    {minimum_temperature:.0f} °C
                </span>

                <span>
                    {maximum_temperature:.0f} °C
                </span>
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

            let searchMarker = null;


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
                                value: station.tmin
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

                const nearestStations =
                    distances.slice(
                        0,
                        Math.min(
                            nearestCount,
                            distances.length
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
                        "<br><b>Interpolated Tmin: " +
                        interpolatedTmin.toFixed(1) +
                        " °C</b>";
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
                        "Tmin: " +
                        interpolatedTmin.toFixed(1) +
                        " °C<br>" +
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


            // =============================================
            // ค่า Tmin ตามเมาส์
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
                            "<b>Interpolated Tmin</b><br>" +
                            "อยู่นอกพื้นที่ประเทศไทย";

                        cursorBox.style.display =
                            "none";

                        return;
                    }}

                    const interpolatedTmin =
                        calculateIDW(
                            latitude,
                            longitude
                        );

                    if (
                        interpolatedTmin === null ||
                        !Number.isFinite(
                            interpolatedTmin
                        )
                    ) {{
                        valueBox.innerHTML =
                            "<b>Interpolated Tmin</b><br>" +
                            "ไม่สามารถคำนวณค่าได้";

                        cursorBox.style.display =
                            "none";

                        return;
                    }}

                    valueBox.innerHTML =
                        "<b>Interpolated Tmin</b><br>" +
                        "Tmin: " +
                        interpolatedTmin.toFixed(1) +
                        " °C<br>" +
                        "Lat: " +
                        latitude.toFixed(4) +
                        "<br>" +
                        "Lon: " +
                        longitude.toFixed(4);

                    cursorBox.textContent =
                        "Tmin: " +
                        interpolatedTmin.toFixed(1) +
                        " °C";

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
            + "</head>",
        )

        map_html = map_html.replace(
            "</body>",
            title_html
            + search_html
            + legend_html
            + mouse_value_html
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
