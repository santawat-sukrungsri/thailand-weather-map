from pathlib import Path
import base64
import html
import io
import json

import folium
import geopandas as gpd
import matplotlib

# ใช้ Matplotlib สำหรับเว็บโดยไม่เปิดหน้าต่างกราฟ
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
# 2. ตำแหน่งไฟล์
# =========================================================

# main.py อยู่ใน:
# C:\Pam\TestMap\app\main.py
#
# BASE_DIR จึงเป็น:
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
# 3. ฟังก์ชันค้นหาชื่อคอลัมน์
# =========================================================

def normalize_column_name(name: str) -> str:
    """
    ปรับชื่อคอลัมน์ให้อยู่ในรูปแบบเดียวกัน
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
    ค้นหาคอลัมน์จากรายชื่อที่เป็นไปได้
    """

    column_lookup = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        normalized_name = normalize_column_name(
            possible_name
        )

        if normalized_name in column_lookup:
            return column_lookup[normalized_name]

    raise ValueError(
        "ไม่พบคอลัมน์ที่ต้องการ\n"
        f"ชื่อที่ค้นหา: {possible_names}\n"
        f"คอลัมน์ใน Excel: {list(dataframe.columns)}"
    )


# =========================================================
# 4. อ่านข้อมูลสถานีจาก Excel
# =========================================================

def load_station_data() -> pd.DataFrame:
    """
    อ่านชื่อสถานี Lat Lon และ Tmin จาก Excel
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
            "MinimumTemperature",
            "MinTemperature",
            "Min Temp",
            "อุณหภูมิต่ำสุด",
        ],
    )

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

    station_data = pd.DataFrame(
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
        station_data["station"] = (
            dataframe[station_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        station_data["station"] = "Weather Station"

    station_data = station_data.dropna(
        subset=[
            "longitude",
            "latitude",
            "tmin",
        ]
    )

    # กรองพิกัดผิดปกติ
    station_data = station_data[
        station_data["longitude"].between(96, 107)
        & station_data["latitude"].between(4, 22)
    ]

    station_data = station_data.reset_index(
        drop=True
    )

    if station_data.empty:
        raise ValueError(
            "ไม่พบข้อมูลสถานีที่สามารถนำมาสร้างแผนที่ได้"
        )

    return station_data


# =========================================================
# 5. อ่านขอบเขตประเทศไทย
# =========================================================

def load_thailand_boundary() -> gpd.GeoDataFrame:
    """
    อ่าน thailand.geojson และปรับ CRS เป็น WGS 84
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

    # แก้ geometry ที่อาจไม่สมบูรณ์
    thailand["geometry"] = (
        thailand.geometry.buffer(0)
    )

    thailand = thailand[
        thailand.geometry.notna()
        & ~thailand.geometry.is_empty
    ].copy()

    if thailand.empty:
        raise ValueError(
            "ไม่พบ geometry ที่ใช้งานได้ใน thailand.geojson"
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
# 7. สร้าง Mask เฉพาะประเทศไทย
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
# 8. สร้างภาพเฉดสีแบบโปร่งใส
# =========================================================

def create_temperature_overlay():
    """
    สร้างภาพ Tmin interpolation สำหรับวางบน Folium
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

    # ใช้ขอบเขตจริงจากไฟล์ประเทศไทย
    minimum_lon, minimum_lat, maximum_lon, maximum_lat = (
        thailand.total_bounds
    )

    # เพิ่มขอบเล็กน้อย
    padding = 0.03

    minimum_lon -= padding
    maximum_lon += padding
    minimum_lat -= padding
    maximum_lat += padding

    # ความละเอียดของกริด
    # ลดลงได้หากเครื่องประมวลผลช้า
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
# 9. หน้าแรก
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
                แผนที่อุณหภูมิต่ำสุดแบบโต้ตอบ
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
# 10. หน้าแผนที่ Interactive
# =========================================================

@app.get(
    "/map",
    response_class=HTMLResponse,
)
def show_map() -> HTMLResponse:
    """
    แสดงแผนที่โลกแบบซูมเข้า–ออกได้

    เมาส์อยู่ในประเทศไทย:
    แสดงค่า Tmin จาก IDW

    เมาส์อยู่นอกประเทศไทย:
    ไม่แสดงค่า Tmin
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

        # -------------------------------------------------
        # สร้างแผนที่
        # -------------------------------------------------

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

        # -------------------------------------------------
        # เพิ่มเฉดสี Tmin
        # -------------------------------------------------

        ImageOverlay(
            image=image_url,
            bounds=image_bounds,
            opacity=0.72,
            name="Interpolated Tmin",
            interactive=False,
            cross_origin=False,
            zindex=2,
        ).add_to(weather_map)

        # -------------------------------------------------
        # เพิ่มเส้นขอบประเทศไทย
        # -------------------------------------------------

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

        # -------------------------------------------------
        # เพิ่มจุดสถานี
        # -------------------------------------------------

        station_group = folium.FeatureGroup(
            name="Weather stations",
            show=True,
        )

        for _, station in station_data.iterrows():

            station_name = str(
                station["station"]
            ).strip()

            if not station_name:
                station_name = "Weather Station"

            safe_station_name = html.escape(
                station_name
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

        minimum_lon, minimum_lat, maximum_lon, maximum_lat = (
            thailand.total_bounds
        )

        weather_map.fit_bounds(
            [
                [minimum_lat, minimum_lon],
                [maximum_lat, maximum_lon],
            ]
        )

        # -------------------------------------------------
        # เตรียมข้อมูล JavaScript
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

        # ชื่อตัวแปร JavaScript ของแผนที่ Folium
        map_variable_name = weather_map.get_name()

        # -------------------------------------------------
        # กล่องชื่อแผนที่
        # -------------------------------------------------

        title_html = """
        <div style="
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            padding: 9px 18px;
            background: rgba(255,255,255,0.94);
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

        # -------------------------------------------------
        # Legend
        # -------------------------------------------------

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

        # -------------------------------------------------
        # กล่องแสดงค่า
        # -------------------------------------------------

        mouse_value_html = """
        <div id="mouse-tmin-box" style="
            position: fixed;
            bottom: 35px;
            right: 35px;
            z-index: 9999;
            min-width: 220px;
            padding: 12px;
            background: rgba(255,255,255,0.96);
            border: 2px solid #555;
            border-radius: 7px;
            font-family: Arial;
            font-size: 14px;
            line-height: 1.5;
            box-shadow: 0 1px 7px
                rgba(0,0,0,0.35);
        ">
            <b>Interpolated Tmin</b><br>
            เลื่อนเมาส์ภายในประเทศไทย
        </div>

        <div id="cursor-tmin-box" style="
            display: none;
            position: fixed;
            z-index: 10000;
            pointer-events: none;
            padding: 6px 9px;
            background: rgba(0,0,0,0.82);
            color: white;
            border-radius: 5px;
            font-family: Arial;
            font-size: 13px;
            white-space: nowrap;
        ">
        </div>
        """

        # -------------------------------------------------
        # JavaScript
        #
        # จุดสำคัญ:
        # สคริปต์ชุดนี้จะถูกใส่ก่อน </html>
        # หลังจาก Folium สร้างตัวแปรแผนที่แล้ว
        # -------------------------------------------------

        mouse_script = f"""
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


            // =============================================
            // ตรวจจุดภายในวงแหวน Polygon
            // =============================================

            function pointInRing(
                longitude,
                latitude,
                ring
            ) {{
                let inside = false;

                for (
                    let index = 0,
                        previousIndex = ring.length - 1;
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


            // =============================================
            // ตรวจ Polygon
            // Ring แรกคือพื้นที่หลัก
            // Ring ถัดไปคือรูภายในพื้นที่
            // =============================================

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


            // =============================================
            // ตรวจว่าอยู่ในประเทศไทยหรือไม่
            // รองรับ Polygon และ MultiPolygon
            // =============================================

            function pointInsideThailand(
                latitude,
                longitude
            ) {{
                const features =
                    thailandGeoJSON.features || [];

                for (
                    const feature of features
                ) {{
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
            // คำนวณ IDW
            // ใช้ power = 2 และสถานีใกล้สุด 8 สถานี
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

                            const distance =
                                Math.sqrt(
                                    dx * dx +
                                    dy * dy
                                );

                            return {{
                                distance: distance,
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
                            weight *
                            item.value;

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
            // เมื่อเลื่อนเมาส์บนแผนที่
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

                    const isInsideThailand =
                        pointInsideThailand(
                            latitude,
                            longitude
                        );

                    if (!isInsideThailand) {{

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

                    cursorBox.innerHTML =
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
                            )
                            + "px";

                        cursorBox.style.top =
                            (
                                originalEvent.clientY +
                                14
                            )
                            + "px";
                    }}
                }}
            );


            // =============================================
            // เมื่อเมาส์ออกจากแผนที่
            // =============================================

            mapObject.on(
                "mouseout",
                function() {{

                    valueBox.innerHTML =
                        "<b>Interpolated Tmin</b><br>" +
                        "เลื่อนเมาส์ภายในประเทศไทย";

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

        map_html = weather_map.get_root().render()

        # กล่องต่าง ๆ ใส่ใน body
        map_html = map_html.replace(
            "</body>",
            title_html
            + legend_html
            + mouse_value_html
            + "</body>",
        )

        # JavaScript ต้องใส่หลัง JavaScript ของ Folium
        # จึงวางไว้ก่อน </html>
        map_html = map_html.replace(
            "</html>",
            mouse_script
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

        page_html = f"""
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
                    overflow-x: auto;
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
                    padding: 10px 16px;
                    margin: 5px;
                    background: #2874b2;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
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
            content=page_html
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