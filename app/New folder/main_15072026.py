from pathlib import Path
import base64
import io

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from folium.raster_layers import ImageOverlay
from scipy.spatial import cKDTree
from shapely import intersects, points


# =========================================================
# 1. สร้าง FastAPI
# =========================================================

app = FastAPI(
    title="Thailand Minimum Temperature Map",
    description="Interactive interpolated Tmin map",
)


# =========================================================
# 2. กำหนดตำแหน่งไฟล์
# =========================================================

# main.py อยู่ใน:
# C:\Pam\TestMap\app\main.py
#
# ดังนั้น parent.parent จะย้อนกลับไปที่:
# C:\Pam\TestMap

BASE_DIR = Path(__file__).resolve().parent.parent

EXCEL_FILE = (
    BASE_DIR
    / "data"
    / "TMD_WeatherToday_Station_LatLon_Temp_Rain_20260709.xlsx"
)

THAILAND_FILE = (
    BASE_DIR
    / "data"
    / "thailand.geojson"
)


# =========================================================
# 3. ฟังก์ชันปรับชื่อคอลัมน์
# =========================================================

def normalize_column_name(name: str) -> str:
    """
    ปรับชื่อคอลัมน์เพื่อให้ค้นหาได้ง่ายขึ้น
    เช่น ลบช่องว่างและเปลี่ยนเป็นตัวพิมพ์เล็ก
    """

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )


def find_column(
    dataframe: pd.DataFrame,
    possible_names: list[str],
) -> str:
    """
    ค้นหาคอลัมน์จากชื่อที่เป็นไปได้
    """

    column_lookup = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for name in possible_names:
        normalized_name = normalize_column_name(name)

        if normalized_name in column_lookup:
            return column_lookup[normalized_name]

    raise ValueError(
        "ไม่พบคอลัมน์ที่ต้องการ\n"
        f"ชื่อที่ค้นหา: {possible_names}\n"
        f"คอลัมน์ที่มีใน Excel: {list(dataframe.columns)}"
    )


# =========================================================
# 4. อ่านข้อมูลสถานีจาก Excel
# =========================================================

def load_station_data() -> pd.DataFrame:
    """
    อ่าน Lat, Lon, Tmin และชื่อสถานีจาก Excel
    """

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Excel:\n{EXCEL_FILE}"
        )

    dataframe = pd.read_excel(
        EXCEL_FILE,
        engine="openpyxl",
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    longitude_column = find_column(
        dataframe,
        [
            "Lon",
            "Longitude",
            "LONG",
            "ลองจิจูด",
        ],
    )

    latitude_column = find_column(
        dataframe,
        [
            "Lat",
            "Latitude",
            "LAT",
            "ละติจูด",
        ],
    )

    tmin_column = find_column(
        dataframe,
        [
            "Tmin (°C)",
            "Tmin",
            "TMIN",
            "Minimum Temperature",
            "MinTemperature",
            "MinimumTemperature",
            "Min Temp",
            "อุณหภูมิต่ำสุด",
        ],
    )

    station_column = None

    try:
        station_column = find_column(
            dataframe,
            [
                "Station English",
                "Station Thai",
                "Station",
                "Station Name",
                "StationName",
                "Name",
                "ชื่อสถานี",
                "สถานี",
            ],
        )
    except ValueError:
        station_column = None

    selected_data = pd.DataFrame(
        {
            "longitude": pd.to_numeric(
                dataframe[longitude_column],
                errors="coerce",
            ),
            "latitude": pd.to_numeric(
                dataframe[latitude_column],
                errors="coerce",
            ),
            "tmin": pd.to_numeric(
                dataframe[tmin_column],
                errors="coerce",
            ),
        }
    )

    if station_column is not None:
        selected_data["station"] = (
            dataframe[station_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        selected_data["station"] = "Weather Station"

    selected_data = selected_data.dropna(
        subset=[
            "longitude",
            "latitude",
            "tmin",
        ]
    )

    # กรองพิกัดที่ผิดปกติ
    selected_data = selected_data[
        selected_data["longitude"].between(96, 107)
        & selected_data["latitude"].between(4, 22)
    ]

    selected_data = selected_data.reset_index(
        drop=True
    )

    if selected_data.empty:
        raise ValueError(
            "ไม่พบข้อมูลสถานีที่สามารถนำมาสร้างแผนที่ได้"
        )

    return selected_data


# =========================================================
# 5. อ่านขอบเขตประเทศไทย
# =========================================================

def load_thailand_boundary() -> gpd.GeoDataFrame:
    """
    อ่านไฟล์ thailand.geojson
    และแปลง CRS เป็น EPSG:4326
    """

    if not THAILAND_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ขอบเขตประเทศไทย:\n{THAILAND_FILE}"
        )

    thailand = gpd.read_file(
        THAILAND_FILE
    )

    if thailand.empty:
        raise ValueError(
            "ไฟล์ thailand.geojson ไม่มีข้อมูลพื้นที่"
        )

    if thailand.crs is None:
        thailand = thailand.set_crs(
            epsg=4326
        )
    else:
        thailand = thailand.to_crs(
            epsg=4326
        )

    # แก้ geometry ที่อาจมีปัญหา
    thailand["geometry"] = (
        thailand.geometry.buffer(0)
    )

    return thailand


# =========================================================
# 6. IDW Interpolation
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
    ทำ Inverse Distance Weighting หรือ IDW
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

    k = min(
        nearest_points,
        len(station_coordinates),
    )

    distances, indexes = tree.query(
        grid_coordinates,
        k=k,
    )

    if k == 1:
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
# 7. สร้าง Mask เฉพาะประเทศไทย
# =========================================================

def create_thailand_mask(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    thailand: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    True  = อยู่ในประเทศไทย
    False = อยู่นอกประเทศไทย
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
# 8. สร้างภาพเฉดสีแบบโปร่งใส
# =========================================================

def create_temperature_overlay():
    """
    สร้างภาพ interpolation โปร่งใส
    เพื่อวางบนแผนที่ Folium
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

    # ขอบเขตภาพครอบคลุมประเทศไทย
    minimum_lon = 97.0
    maximum_lon = 106.0
    minimum_lat = 5.0
    maximum_lat = 21.0

    # ความละเอียดของกริด
    # ถ้าเครื่องช้า ลดลงได้ เช่น 120 × 220
    grid_width = 180
    grid_height = 320

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
        21,
    )

    figure, axis = plt.subplots(
        figsize=(8, 12),
        dpi=100,
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

    bounds = [
        [minimum_lat, minimum_lon],
        [maximum_lat, maximum_lon],
    ]

    return (
        image_url,
        bounds,
        station_data,
        thailand,
        minimum_temperature,
        maximum_temperature,
    )


# =========================================================
# 9. หน้าแรก
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home() -> HTMLResponse:
    """
    หน้าแรกของโปรแกรม
    """

    html = """
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
                background: #173f6d;
                color: white;
                padding: 20px;
            }

            .header h1 {
                margin: 0;
                font-size: 25px;
            }

            .header p {
                margin: 8px 0 0;
            }

            .content {
                padding: 28px;
            }

            .button {
                display: inline-block;
                margin: 6px;
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
                แผนที่อุณหภูมิต่ำสุดแบบซูมเข้า–ออกได้
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
        content=html
    )


# =========================================================
# 10. หน้าแผนที่แบบ Interactive
# =========================================================

@app.get(
    "/map",
    response_class=HTMLResponse,
)
def show_map() -> HTMLResponse:
    """
    แสดงแผนที่โลกแบบซูมเข้า–ออกได้
    และแสดงเฉดสีเฉพาะประเทศไทย
    """

    try:
        (
            image_url,
            bounds,
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

        # แผนที่พื้นฐาน
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            control=True,
            show=True,
        ).add_to(weather_map)

        folium.TileLayer(
            tiles="CartoDB positron",
            name="แผนที่พื้นอ่อน",
            control=True,
            show=False,
        ).add_to(weather_map)

        folium.TileLayer(
            tiles="CartoDB dark_matter",
            name="แผนที่พื้นเข้ม",
            control=True,
            show=False,
        ).add_to(weather_map)

        # ภาพเฉดสี Tmin
        ImageOverlay(
            image=image_url,
            bounds=bounds,
            opacity=0.72,
            name="Interpolated Tmin",
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(weather_map)

        # เส้นขอบประเทศไทย
        folium.GeoJson(
            data=thailand.to_json(),
            name="Thailand boundary",
            style_function=lambda feature: {
                "fillColor": "transparent",
                "color": "black",
                "weight": 1.4,
                "fillOpacity": 0,
            },
        ).add_to(weather_map)

        # กลุ่มสถานี
        station_group = folium.FeatureGroup(
            name="Weather stations",
            show=True,
        )

        for _, station in station_data.iterrows():
            station_name = station["station"]

            if not station_name:
                station_name = "Weather Station"

            popup_html = f"""
            <div style="
                font-family: Arial;
                font-size: 14px;
                min-width: 180px;
            ">
                <b>{station_name}</b><br>
                Latitude:
                {station["latitude"]:.4f}<br>
                Longitude:
                {station["longitude"]:.4f}<br>
                Tmin:
                {station["tmin"]:.1f} °C
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
                    f'{station_name}: '
                    f'{station["tmin"]:.1f} °C'
                ),
                popup=folium.Popup(
                    popup_html,
                    max_width=280,
                ),
            ).add_to(station_group)

        station_group.add_to(weather_map)

        folium.LayerControl(
            collapsed=False
        ).add_to(weather_map)

        weather_map.fit_bounds(
            [
                [5.0, 97.0],
                [21.0, 106.0],
            ]
        )

        map_html = weather_map.get_root().render()

        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 35px;
            left: 35px;
            z-index: 9999;
            width: 220px;
            padding: 12px;
            background: rgba(255,255,255,0.95);
            border: 2px solid #555;
            border-radius: 7px;
            font-family: Arial;
            font-size: 13px;
            box-shadow: 0 1px 7px
                rgba(0,0,0,0.35);
        ">
            <b>Minimum Temperature</b>

            <div style="
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
            "></div>

            <div style="
                display: flex;
                justify-content: space-between;
            ">
                <span>
                    {minimum_temperature:.0f} °C
                </span>

                <span>
                    {maximum_temperature:.0f} °C
                </span>
            </div>
        </div>
        """

        title_html = """
        <div style="
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            padding: 9px 18px;
            background: rgba(255,255,255,0.92);
            border-radius: 7px;
            font-family: Arial;
            font-weight: bold;
            font-size: 17px;
            box-shadow: 0 1px 6px
                rgba(0,0,0,0.35);
        ">
            Interpolated Minimum Temperature in Thailand
        </div>
        """

        map_html = map_html.replace(
            "</body>",
            title_html
            + legend_html
            + "</body>",
        )

        return HTMLResponse(
            content=map_html
        )

    except Exception as error:
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="th">
            <head>
                <meta charset="UTF-8">
                <title>เกิดข้อผิดพลาด</title>
            </head>

            <body style="
                background: #eeeeee;
                font-family: Arial;
                padding: 30px;
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
                    ">{str(error)}</pre>

                    <p>
                        ตรวจสอบชื่อไฟล์ Excel,
                        ชื่อคอลัมน์ และไฟล์
                        thailand.geojson
                    </p>
                </div>
            </body>
            </html>
            """,
            status_code=500,
        )


# =========================================================
# 11. หน้าแสดงข้อมูลสถานี
# =========================================================

@app.get(
    "/stations",
    response_class=HTMLResponse,
)
def show_stations() -> HTMLResponse:
    """
    แสดงข้อมูลสถานีเป็นตาราง
    """

    try:
        station_data = load_station_data()

        table_html = station_data.to_html(
            index=False,
            border=0,
            classes="station-table",
            float_format=lambda value: f"{value:.3f}",
        )

        html = f"""
        <!DOCTYPE html>
        <html lang="th">
        <head>
            <meta charset="UTF-8">

            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
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
                    box-shadow: 0 2px 10px
                        rgba(0,0,0,0.12);
                }}

                .station-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}

                .station-table th,
                .station-table td {{
                    padding: 8px;
                    border: 1px solid #cccccc;
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
                    margin-bottom: 14px;
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

        return HTMLResponse(
            content=html
        )

    except Exception as error:
        return HTMLResponse(
            content=f"""
            <h2>
                ไม่สามารถอ่านข้อมูลสถานีได้
            </h2>

            <pre>
                {str(error)}
            </pre>
            """,
            status_code=500,
        )