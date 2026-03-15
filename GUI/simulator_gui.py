import pygame
import math
import random
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
import shapefile
import numpy as np

pygame.init()


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 950
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (10, 15, 30)

LAND_COLOR = (30, 50, 35)
OCEAN_COLOR = (12, 25, 50)
COAST_COLOR = (40, 65, 45)

MAP_OFFSET_X = 20
MAP_OFFSET_Y = 50
MAP_WIDTH = 1150
MAP_HEIGHT = 650
INFO_PANEL_X = MAP_OFFSET_X + MAP_WIDTH + 20

WEATHER_GRADIENTS = {
    'clear': [(255, 250, 150), (255, 230, 120), (255, 210, 80)],
    'cloudy': [(170, 175, 185), (150, 155, 165), (130, 135, 145)],
    'rain': [(70, 110, 190), (55, 95, 175), (40, 80, 160)],
    'storm': [(110, 50, 140), (90, 35, 120), (70, 20, 100)],
    'snow': [(210, 230, 255), (195, 215, 245), (180, 200, 235)],
    'fog': [(135, 140, 145), (125, 130, 135), (115, 120, 125)],
    'wind': [(90, 170, 150), (75, 155, 135), (60, 140, 120)],
}

CITY_DATABASE = [
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'country': 'US'},
    {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437, 'country': 'US'},
    {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298, 'country': 'US'},
    {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698, 'country': 'US'},
    {'name': 'Phoenix', 'lat': 33.4484, 'lon': -112.0740, 'country': 'US'},
    {'name': 'Philadelphia', 'lat': 39.9526, 'lon': -75.1652, 'country': 'US'},
    {'name': 'San Antonio', 'lat': 29.4241, 'lon': -98.4936, 'country': 'US'},
    {'name': 'San Diego', 'lat': 32.7157, 'lon': -117.1611, 'country': 'US'},
    {'name': 'Dallas', 'lat': 32.7767, 'lon': -96.7970, 'country': 'US'},
    {'name': 'San Jose', 'lat': 37.3382, 'lon': -121.8863, 'country': 'US'},
    {'name': 'Austin', 'lat': 30.2672, 'lon': -97.7431, 'country': 'US'},
    {'name': 'Jacksonville', 'lat': 30.3322, 'lon': -81.6557, 'country': 'US'},
    {'name': 'Fort Worth', 'lat': 32.7555, 'lon': -97.3308, 'country': 'US'},
    {'name': 'Columbus', 'lat': 39.9612, 'lon': -83.0000, 'country': 'US'},
    {'name': 'Charlotte', 'lat': 35.2271, 'lon': -80.8431, 'country': 'US'},
    {'name': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194, 'country': 'US'},
    {'name': 'Indianapolis', 'lat': 39.7684, 'lon': -86.1581, 'country': 'US'},
    {'name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321, 'country': 'US'},
    {'name': 'Denver', 'lat': 39.7392, 'lon': -104.9903, 'country': 'US'},
    {'name': 'Washington', 'lat': 38.9072, 'lon': -77.0369, 'country': 'US'},
    {'name': 'Boston', 'lat': 42.3601, 'lon': -71.0589, 'country': 'US'},
    {'name': 'El Paso', 'lat': 31.7619, 'lon': -106.4850, 'country': 'US'},
    {'name': 'Nashville', 'lat': 36.1627, 'lon': -86.7816, 'country': 'US'},
    {'name': 'Detroit', 'lat': 42.3314, 'lon': -83.0458, 'country': 'US'},
    {'name': 'Portland', 'lat': 45.5152, 'lon': -122.6784, 'country': 'US'},
    {'name': 'Las Vegas', 'lat': 36.1699, 'lon': -115.1398, 'country': 'US'},
    {'name': 'Memphis', 'lat': 35.1495, 'lon': -90.0490, 'country': 'US'},
    {'name': 'Louisville', 'lat': 38.2527, 'lon': -85.7585, 'country': 'US'},
    {'name': 'Baltimore', 'lat': 39.2904, 'lon': -76.6122, 'country': 'US'},
    {'name': 'Milwaukee', 'lat': 43.0389, 'lon': -87.9065, 'country': 'US'},
    {'name': 'Toronto', 'lat': 43.6532, 'lon': -79.3832, 'country': 'CA'},
    {'name': 'Vancouver', 'lat': 49.2827, 'lon': -123.1207, 'country': 'CA'},
    {'name': 'Montreal', 'lat': 45.5017, 'lon': -73.5673, 'country': 'CA'},
    {'name': 'Calgary', 'lat': 51.0447, 'lon': -114.0719, 'country': 'CA'},
    {'name': 'Edmonton', 'lat': 53.5461, 'lon': -113.4938, 'country': 'CA'},
    {'name': 'Ottawa', 'lat': 45.4215, 'lon': -75.6972, 'country': 'CA'},
    {'name': 'Winnipeg', 'lat': 49.8951, 'lon': -97.1384, 'country': 'CA'},
    {'name': 'Quebec City', 'lat': 46.8139, 'lon': -71.2082, 'country': 'CA'},
    {'name': 'Mexico City', 'lat': 19.4326, 'lon': -99.1332, 'country': 'MX'},
    {'name': 'Guadalajara', 'lat': 20.6597, 'lon': -103.3496, 'country': 'MX'},
    {'name': 'Monterrey', 'lat': 25.6866, 'lon': -100.3161, 'country': 'MX'},
    {'name': 'Sao Paulo', 'lat': -23.5505, 'lon': -46.6333, 'country': 'BR'},
    {'name': 'Rio de Janeiro', 'lat': -22.9068, 'lon': -43.1729, 'country': 'BR'},
    {'name': 'Buenos Aires', 'lat': -34.6037, 'lon': -58.3816, 'country': 'AR'},
    {'name': 'Lima', 'lat': -12.0464, 'lon': -77.0428, 'country': 'PE'},
    {'name': 'Bogota', 'lat': 4.7110, 'lon': -74.0721, 'country': 'CO'},
    {'name': 'Santiago', 'lat': -33.4489, 'lon': -70.6693, 'country': 'CL'},
    {'name': 'Caracas', 'lat': 10.4806, 'lon': -66.9036, 'country': 'VE'},
    {'name': 'London', 'lat': 51.5074, 'lon': -0.1278, 'country': 'UK'},
    {'name': 'Paris', 'lat': 48.8566, 'lon': 2.3522, 'country': 'FR'},
    {'name': 'Berlin', 'lat': 52.5200, 'lon': 13.4050, 'country': 'DE'},
    {'name': 'Madrid', 'lat': 40.4168, 'lon': -3.7038, 'country': 'ES'},
    {'name': 'Rome', 'lat': 41.9028, 'lon': 12.4964, 'country': 'IT'},
    {'name': 'Amsterdam', 'lat': 52.3676, 'lon': 4.9041, 'country': 'NL'},
    {'name': 'Vienna', 'lat': 48.2082, 'lon': 16.3738, 'country': 'AT'},
    {'name': 'Brussels', 'lat': 50.8503, 'lon': 4.3517, 'country': 'BE'},
    {'name': 'Warsaw', 'lat': 52.2297, 'lon': 21.0122, 'country': 'PL'},
    {'name': 'Prague', 'lat': 50.0755, 'lon': 14.4378, 'country': 'CZ'},
    {'name': 'Athens', 'lat': 37.9838, 'lon': 23.7275, 'country': 'GR'},
    {'name': 'Lisbon', 'lat': 38.7223, 'lon': -9.1393, 'country': 'PT'},
    {'name': 'Stockholm', 'lat': 59.3293, 'lon': 18.0686, 'country': 'SE'},
    {'name': 'Oslo', 'lat': 59.9139, 'lon': 10.7522, 'country': 'NO'},
    {'name': 'Copenhagen', 'lat': 55.6761, 'lon': 12.5683, 'country': 'DK'},
    {'name': 'Helsinki', 'lat': 60.1699, 'lon': 24.9384, 'country': 'FI'},
    {'name': 'Dublin', 'lat': 53.3498, 'lon': -6.2603, 'country': 'IE'},
    {'name': 'Moscow', 'lat': 55.7558, 'lon': 37.6173, 'country': 'RU'},
    {'name': 'St. Petersburg', 'lat': 59.9311, 'lon': 30.3609, 'country': 'RU'},
    {'name': 'Istanbul', 'lat': 41.0082, 'lon': 28.9784, 'country': 'TR'},
    {'name': 'Cairo', 'lat': 30.0444, 'lon': 31.2357, 'country': 'EG'},
    {'name': 'Cape Town', 'lat': -33.9249, 'lon': 18.4241, 'country': 'ZA'},
    {'name': 'Johannesburg', 'lat': -26.2041, 'lon': 28.0473, 'country': 'ZA'},
    {'name': 'Nairobi', 'lat': -1.2921, 'lon': 36.8219, 'country': 'KE'},
    {'name': 'Lagos', 'lat': 6.5244, 'lon': 3.3792, 'country': 'NG'},
    {'name': 'Casablanca', 'lat': 33.5731, 'lon': -7.5898, 'country': 'MA'},
    {'name': 'Dubai', 'lat': 25.2048, 'lon': 55.2708, 'country': 'AE'},
    {'name': 'Riyadh', 'lat': 24.7136, 'lon': 46.6753, 'country': 'SA'},
    {'name': 'Tel Aviv', 'lat': 32.0853, 'lon': 34.7818, 'country': 'IL'},
    {'name': 'Tehran', 'lat': 35.6892, 'lon': 51.3890, 'country': 'IR'},
    {'name': 'Delhi', 'lat': 28.7041, 'lon': 77.1025, 'country': 'IN'},
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'country': 'IN'},
    {'name': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639, 'country': 'IN'},
    {'name': 'Chennai', 'lat': 13.0827, 'lon': 80.2707, 'country': 'IN'},
    {'name': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946, 'country': 'IN'},
    {'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011, 'country': 'PK'},
    {'name': 'Lahore', 'lat': 31.5204, 'lon': 74.3587, 'country': 'PK'},
    {'name': 'Dhaka', 'lat': 23.8103, 'lon': 90.4125, 'country': 'BD'},
    {'name': 'Beijing', 'lat': 39.9042, 'lon': 116.4074, 'country': 'CN'},
    {'name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737, 'country': 'CN'},
    {'name': 'Guangzhou', 'lat': 23.1291, 'lon': 113.2644, 'country': 'CN'},
    {'name': 'Shenzhen', 'lat': 22.5431, 'lon': 114.0579, 'country': 'CN'},
    {'name': 'Chengdu', 'lat': 30.5728, 'lon': 104.0668, 'country': 'CN'},
    {'name': 'Hong Kong', 'lat': 22.3193, 'lon': 114.1694, 'country': 'HK'},
    {'name': 'Taipei', 'lat': 25.0330, 'lon': 121.5654, 'country': 'TW'},
    {'name': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'country': 'JP'},
    {'name': 'Osaka', 'lat': 34.6937, 'lon': 135.5023, 'country': 'JP'},
    {'name': 'Seoul', 'lat': 37.5665, 'lon': 126.9780, 'country': 'KR'},
    {'name': 'Busan', 'lat': 35.1796, 'lon': 129.0756, 'country': 'KR'},
    {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198, 'country': 'SG'},
    {'name': 'Bangkok', 'lat': 13.7563, 'lon': 100.5018, 'country': 'TH'},
    {'name': 'Kuala Lumpur', 'lat': 3.1390, 'lon': 101.6869, 'country': 'MY'},
    {'name': 'Jakarta', 'lat': -6.2088, 'lon': 106.8456, 'country': 'ID'},
    {'name': 'Manila', 'lat': 14.5995, 'lon': 120.9842, 'country': 'PH'},
    {'name': 'Hanoi', 'lat': 21.0285, 'lon': 105.8542, 'country': 'VN'},
    {'name': 'Ho Chi Minh', 'lat': 10.8231, 'lon': 106.6297, 'country': 'VN'},
    {'name': 'Sydney', 'lat': -33.8688, 'lon': 151.2093, 'country': 'AU'},
    {'name': 'Melbourne', 'lat': -37.8136, 'lon': 144.9631, 'country': 'AU'},
    {'name': 'Brisbane', 'lat': -27.4698, 'lon': 153.0251, 'country': 'AU'},
    {'name': 'Perth', 'lat': -31.9505, 'lon': 115.8605, 'country': 'AU'},
    {'name': 'Auckland', 'lat': -36.8509, 'lon': 174.7645, 'country': 'NZ'},
    {'name': 'Wellington', 'lat': -41.2865, 'lon': 174.7762, 'country': 'NZ'},
]


@dataclass
class WeatherData:
    temp: float
    humidity: int
    wind_speed: float
    wind_dir: float
    condition: str
    pressure: float
    visibility: float
    precipitation: float
    cloud_cover: int


class LandData:
    def __init__(self):
        self.polygons: List[List[Tuple[float, float]]] = []
        self.load_shapefile()
    
    def load_shapefile(self):
        try:
            sf = shapefile.Reader("ne_50m_land/ne_50m_land.shp")
            for shape in sf.shapes():
                if shape.shapeType == shapefile.POLYGON or shape.shapeType == shapefile.POLYLINE:
                    points = [(float(point[0]), float(point[1])) for point in shape.points]
                    if len(points) > 2:
                        self.polygons.append(points)
        except Exception as e:
            print(f"Could not load shapefile: {e}")
            self._load_fallback_data()
    
    def _load_fallback_data(self):
        continents = [
            [(-170.0, 65.0), (-140.0, 72.0), (-100.0, 72.0), (-80.0, 65.0), (-75.0, 45.0), (-55.0, 45.0), (-60.0, 25.0), (-85.0, 10.0), (-120.0, 25.0), (-125.0, 35.0), (-115.0, 32.0), (-105.0, 25.0), (-95.0, 20.0), (-85.0, 20.0), (-82.0, 25.0), (-80.0, 30.0), (-75.0, 35.0), (-70.0, 42.0), (-68.0, 45.0), (-55.0, 48.0), (-68.0, 55.0), (-80.0, 50.0), (-95.0, 50.0), (-105.0, 45.0), (-115.0, 35.0), (-125.0, 40.0), (-135.0, 55.0), (-150.0, 60.0), (-170.0, 65.0)],
            [(-82.0, 10.0), (-75.0, 5.0), (-55.0, -5.0), (-40.0, -5.0), (-35.0, -10.0), (-35.0, -25.0), (-55.0, -55.0), (-70.0, -55.0), (-75.0, -45.0), (-75.0, -30.0), (-80.0, -5.0), (-82.0, 10.0)],
            [(-10.0, 35.0), (5.0, 45.0), (10.0, 45.0), (30.0, 45.0), (30.0, 55.0), (25.0, 60.0), (30.0, 70.0), (60.0, 70.0), (70.0, 55.0), (60.0, 35.0), (50.0, 35.0), (40.0, 35.0), (35.0, 40.0), (25.0, 35.0), (15.0, 40.0), (5.0, 42.0), (-10.0, 35.0)],
            [(-20.0, 35.0), (-5.0, 42.0), (10.0, 45.0), (15.0, 42.0), (25.0, 35.0), (35.0, 35.0), (45.0, 30.0), (50.0, 15.0), (55.0, 15.0), (60.0, 20.0), (50.0, 15.0), (40.0, 5.0), (35.0, -5.0), (35.0, -15.0), (30.0, -35.0), (25.0, -35.0), (15.0, -30.0), (10.0, -5.0), (-10.0, 5.0), (-15.0, 15.0), (-20.0, 35.0)],
            [(60.0, 35.0), (75.0, 45.0), (90.0, 50.0), (105.0, 55.0), (120.0, 50.0), (135.0, 50.0), (145.0, 45.0), (155.0, 50.0), (165.0, 60.0), (175.0, 65.0), (170.0, 65.0), (160.0, 55.0), (140.0, 50.0), (130.0, 50.0), (120.0, 45.0), (110.0, 35.0), (100.0, 30.0), (90.0, 25.0), (80.0, 15.0), (80.0, 10.0), (75.0, 5.0), (100.0, 10.0), (105.0, 5.0), (120.0, -5.0), (130.0, -5.0), (135.0, -5.0), (140.0, -15.0), (145.0, -35.0), (165.0, -40.0), (175.0, -40.0), (170.0, -35.0), (160.0, -25.0), (150.0, -15.0), (140.0, -5.0), (130.0, 5.0), (120.0, 15.0), (110.0, 25.0), (100.0, 25.0), (90.0, 25.0), (80.0, 25.0), (70.0, 20.0), (60.0, 30.0), (60.0, 35.0)],
            [(140.0, -15.0), (150.0, -10.0), (155.0, -15.0), (155.0, -25.0), (150.0, -40.0), (145.0, -40.0), (135.0, -35.0), (135.0, -25.0), (140.0, -15.0)],
        ]
        self.polygons = continents


class WeatherFront:
    def __init__(self, center_lat: float, center_lon: float,
                 front_type: str, intensity: float, direction: float, speed: float):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.front_type = front_type
        self.intensity = intensity
        self.direction = direction
        self.speed = speed
        self.radius = random.uniform(4, 10)
        self.age = 0
        self.max_age = random.randint(400, 800)
        self.rotation = random.uniform(0, 360)

    def update(self, dt: float):
        rad = math.radians(self.direction)
        self.center_lat += math.cos(rad) * self.speed * dt * 0.08
        self.center_lon += math.sin(rad) * self.speed * dt * 0.15
        self.age += 1
        self.rotation += dt * 30

        if random.random() < 0.02:
            self.radius += random.uniform(-0.3, 0.3)
            self.radius = max(2, min(15, self.radius))

        if random.random() < 0.02:
            self.intensity += random.uniform(-0.03, 0.03)
            self.intensity = max(0.3, min(1.0, self.intensity))

    def is_alive(self) -> bool:
        return self.age < self.max_age and -85 <= self.center_lat <= 85 and -180 <= self.center_lon <= 180


class WeatherAPI:
    def __init__(self):
        self.api_url = "https://api.open-meteo.com/v1/forecast"
        self.cache: Dict[str, Tuple[WeatherData, float]] = {}
        self.cache_duration = 180

    def get_weather(self, lat: float, lon: float, location_key: str) -> WeatherData:
        current_time = time.time()
        if location_key in self.cache:
            cached_data, cache_time = self.cache[location_key]
            if current_time - cache_time < self.cache_duration:
                return cached_data

        try:
            url = (f"{self.api_url}?"
                   f"latitude={lat:.4f}&longitude={lon:.4f}&"
                   f"current=temperature_2m,relative_humidity_2m,weather_code,"
                   f"surface_pressure,wind_speed_10m,wind_direction_10m,"
                   f"visibility,precipitation,cloud_cover&"
                   f"timezone=auto")
            req = urllib.request.Request(url, headers={'User-Agent': 'WeatherGUI/2.0'})
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode())

            current = data.get('current', {})
            weather_code = current.get('weather_code', 0)
            condition = self._code_to_condition(weather_code)

            weather = WeatherData(
                temp=current.get('temperature_2m', 20.0),
                humidity=current.get('relative_humidity_2m', 50),
                wind_speed=current.get('wind_speed_10m', 5.0),
                wind_dir=current.get('wind_direction_10m', 0),
                condition=condition,
                pressure=current.get('surface_pressure', 1013.0),
                visibility=current.get('visibility', 10.0),
                precipitation=current.get('precipitation', 0.0),
                cloud_cover=current.get('cloud_cover', 0)
            )

            self.cache[location_key] = (weather, current_time)
            return weather
        except Exception:
            return self._generate_weather_by_location(lat)

    def _code_to_condition(self, code: int) -> str:
        if code == 0:
            return 'clear'
        elif code in [1, 2, 3]:
            return 'cloudy'
        elif code in [51, 53, 55, 56, 57]:
            return 'fog'
        elif code in [61, 63, 65, 66, 67, 80, 81, 82]:
            return 'rain'
        elif code in [95, 96, 99]:
            return 'storm'
        elif code in [71, 73, 75, 77, 85, 86]:
            return 'snow'
        else:
            return 'cloudy'

    def _generate_weather_by_location(self, lat: float) -> WeatherData:
        abs_lat = abs(lat)
        season_factor = math.cos(math.radians(lat * 2))

        if abs_lat > 60:
            temp = random.uniform(-20, 5)
            conditions = ['snow', 'cloudy', 'storm', 'fog']
            weights = [40, 30, 20, 10]
        elif abs_lat > 45:
            temp = random.uniform(-5, 25) * season_factor
            conditions = ['clear', 'cloudy', 'rain', 'snow']
            weights = [25, 35, 25, 15]
        elif abs_lat > 25:
            temp = random.uniform(10, 35) * season_factor
            conditions = ['clear', 'cloudy', 'rain', 'storm']
            weights = [40, 30, 20, 10]
        elif abs_lat > 10:
            temp = random.uniform(18, 38)
            conditions = ['clear', 'cloudy', 'rain', 'storm']
            weights = [50, 25, 20, 5]
        else:
            temp = random.uniform(22, 38)
            conditions = ['clear', 'cloudy', 'rain', 'storm']
            weights = [55, 20, 20, 5]

        condition = random.choices(conditions, weights=weights)[0]

        if condition == 'rain':
            humidity = random.randint(65, 95)
            pressure = random.uniform(990, 1010)
        elif condition == 'storm':
            humidity = random.randint(75, 100)
            pressure = random.uniform(980, 1000)
        elif condition == 'snow':
            humidity = random.randint(50, 80)
            pressure = random.uniform(1000, 1025)
        else:
            humidity = random.randint(25, 70)
            pressure = random.uniform(1010, 1030)

        return WeatherData(
            temp=temp + random.uniform(-2, 2),
            humidity=humidity,
            wind_speed=random.uniform(0, 50) if condition != 'clear' else random.uniform(0, 15),
            wind_dir=random.uniform(0, 360),
            condition=condition,
            pressure=pressure,
            visibility=random.uniform(2, 20) if condition not in ['storm', 'fog'] else random.uniform(0.5, 5),
            precipitation=random.uniform(0, 25) if condition in ['rain', 'storm'] else random.uniform(0, 3),
            cloud_cover=random.randint(60, 100) if condition != 'clear' else random.randint(0, 30)
        )


class WorldMap:
    def __init__(self, width: int, height: int, offset_x: int, offset_y: int, land_data: LandData, min_lat, max_lat, min_lon, max_lon):
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.land_data = land_data
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.scaled_polygons: List[List[Tuple[int, int]]] = []
        self._scale_polygons()

    def _scale_polygons(self):
        for polygon in self.land_data.polygons:
            scaled = []
            for lon, lat in polygon:
                x, y = self.lat_lon_to_screen(lat, lon)
                scaled.append((x, y))
            if len(scaled) >= 3:
                self.scaled_polygons.append(scaled)

    def lat_lon_to_screen(self, lat: float, lon: float) -> Tuple[int, int]:
        x = self.offset_x + int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.width)
        y = self.offset_y + int((self.max_lat - lat) / (self.max_lat - self.min_lat) * self.height)
        return (x, y)

    def screen_to_lat_lon(self, x: int, y: int) -> Tuple[float, float]:
        lon = self.min_lon + (x - self.offset_x) / self.width * (self.max_lon - self.min_lon)
        lat = self.max_lat - (y - self.offset_y) / self.height * (self.max_lat - self.min_lat)
        return (lat, lon)

    def draw(self, surface: pygame.Surface):
        surface.fill(OCEAN_COLOR, (self.offset_x - 5, self.offset_y - 5, 
                                    self.width + 10, self.height + 10))
        
        for polygon in self.scaled_polygons:
            if len(polygon) >= 3:
                pygame.draw.polygon(surface, LAND_COLOR, polygon)
                pygame.draw.polygon(surface, COAST_COLOR, polygon, 1)

    def draw_grid(self, surface: pygame.Surface):
        grid_color = (40, 50, 70)
        label_color = (80, 100, 130)
        font = pygame.font.Font(None, 18)

        for lat in range(-80, 90, 20):
            x1, y1 = self.lat_lon_to_screen(lat, self.min_lon)
            x2, y2 = self.lat_lon_to_screen(lat, self.max_lon)
            pygame.draw.line(surface, grid_color, (x1, y1), (x2, y2), 1)
            
            label = font.render(f"{lat}°", True, label_color)
            surface.blit(label, (self.offset_x - 30, y1 - 6))

        for lon in range(-180, 190, 30):
            x1, y1 = self.lat_lon_to_screen(self.min_lat, lon)
            x2, y2 = self.lat_lon_to_screen(self.max_lat, lon)
            pygame.draw.line(surface, grid_color, (x1, y1), (x2, y2), 1)
            
            if lon != -180:
                label = font.render(f"{lon}°", True, label_color)
                surface.blit(label, (x1 - 10, self.offset_y + self.height + 5))


class FrontVisualizer:
    def __init__(self):
        self.fronts: List[WeatherFront] = []
        self.create_initial_fronts()

    def create_initial_fronts(self):
        for _ in range(6):
            lat = random.uniform(-60, 70)
            lon = random.uniform(-160, 160)
            conditions = ['rain', 'storm', 'snow']
            weights = [50, 30, 20]
            if abs(lat) > 50:
                weights = [20, 30, 50]
            elif abs(lat) < 25:
                weights = [60, 25, 15]
            front_type = random.choices(conditions, weights=weights)[0]
            self.fronts.append(WeatherFront(
                center_lat=lat,
                center_lon=lon,
                front_type=front_type,
                intensity=random.uniform(0.5, 1.0),
                direction=random.uniform(0, 360),
                speed=random.uniform(0.5, 2.0)
            ))

    def update(self, dt: float):
        for front in self.fronts:
            front.update(dt)
        self.fronts = [f for f in self.fronts if f.is_alive()]
        if random.random() < 0.003:
            self._add_random_front()

    def _add_random_front(self):
        lat = random.uniform(-60, 70)
        lon = random.uniform(-160, 160)
        conditions = ['rain', 'storm', 'snow']
        weights = [50, 30, 20]
        if abs(lat) > 50:
            weights = [20, 30, 50]
        elif abs(lat) < 25:
            weights = [60, 25, 15]
        front_type = random.choices(conditions, weights=weights)[0]
        self.fronts.append(WeatherFront(
            center_lat=lat,
            center_lon=lon,
            front_type=front_type,
            intensity=random.uniform(0.5, 1.0),
            direction=random.uniform(0, 360),
            speed=random.uniform(0.5, 2.0)
        ))

    def draw(self, surface: pygame.Surface, world_map: WorldMap, time: float):
        for front in self.fronts:
            x, y = world_map.lat_lon_to_screen(front.center_lat, front.center_lon)
            gradient = WEATHER_GRADIENTS.get(front.front_type, WEATHER_GRADIENTS['cloudy'])
            max_radius = int(front.radius * 12 * front.intensity)
            pulse = math.sin(time * 2) * 0.15 + 0.85

            for i in range(5):
                r = int(max_radius * (1 - i * 0.18) * pulse)
                if r > 0:
                    alpha = int(70 - i * 12)
                    alpha_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                    color = gradient[min(i, len(gradient) - 1)]
                    pygame.draw.circle(alpha_surface, (*color, alpha), (r, r), r)
                    surface.blit(alpha_surface, (x - r, y - r))

            if front.front_type == 'storm':
                self._draw_storm_animation(surface, x, y, front, time, gradient)
            elif front.front_type == 'rain':
                self._draw_rain_animation(surface, x, y, front, time, gradient)
            elif front.front_type == 'snow':
                self._draw_snow_animation(surface, x, y, front, time, gradient)

    def _draw_storm_animation(self, surface: pygame.Surface, x: int, y: int,
                              front: WeatherFront, time: float, gradient: List):
        radius = int(front.radius * 8)
        for i in range(8):
            angle = math.radians(i * 45 + front.rotation)
            for j in range(3):
                r = radius * (0.4 + j * 0.25)
                px = x + math.cos(angle + j * 0.3) * r
                py = y + math.sin(angle + j * 0.3) * r
                alpha = int(120 * front.intensity * (1 - j * 0.25))
                alpha_surface = pygame.Surface((12, 12), pygame.SRCALPHA)
                color = gradient[min(j + 1, len(gradient) - 1)]
                pygame.draw.circle(alpha_surface, (*color, alpha), (6, 6), 6)
                surface.blit(alpha_surface, (int(px) - 6, int(py) - 6))

    def _draw_rain_animation(self, surface: pygame.Surface, x: int, y: int,
                            front: WeatherFront, time: float, gradient: List):
        for i in range(6):
            offset_y = ((time * 30 + i * 15) % 40) - 20
            alpha = int(80 * front.intensity)
            alpha_surface = pygame.Surface((40, 50), pygame.SRCALPHA)
            pygame.draw.ellipse(alpha_surface, (*gradient[0], alpha), (0, int(offset_y), 40, 50))
            surface.blit(alpha_surface, (x - 20, y - 25))

    def _draw_snow_animation(self, surface: pygame.Surface, x: int, y: int,
                            front: WeatherFront, time: float, gradient: List):
        for i in range(10):
            angle = (i / 10) * math.pi * 2 + time
            dist = 15 + math.sin(time * 2 + i) * 8
            px = x + math.cos(angle) * dist
            py = y + math.sin(angle) * dist
            alpha = int(150 * front.intensity)
            alpha_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.circle(alpha_surface, (*gradient[0], alpha), (5, 5), 4)
            surface.blit(alpha_surface, (int(px) - 5, int(py) - 5))


class ParticleSystem:
    def __init__(self):
        self.particles: List[Dict] = []

    def add_particles(self, x: int, y: int, condition: str, intensity: float):
        if condition == 'rain':
            for _ in range(int(intensity * 2)):
                self.particles.append({
                    'x': x + random.randint(-15, 15),
                    'y': y + random.randint(-15, 15),
                    'vx': random.uniform(-0.3, 0.3),
                    'vy': random.uniform(2, 4),
                    'life': random.randint(20, 40),
                    'type': 'rain',
                    'alpha': random.randint(100, 180)
                })
        elif condition == 'snow':
            for _ in range(int(intensity * 1.5)):
                self.particles.append({
                    'x': x + random.randint(-20, 20),
                    'y': y + random.randint(-20, 20),
                    'vx': math.sin(random.uniform(0, math.pi * 2)) * 0.3,
                    'vy': random.uniform(0.4, 1.2),
                    'life': random.randint(40, 80),
                    'type': 'snow',
                    'size': random.uniform(1.5, 3),
                    'angle': random.uniform(0, math.pi * 2),
                    'alpha': random.randint(180, 255)
                })

    def update(self):
        for p in self.particles:
            if p['type'] == 'rain':
                p['y'] += p['vy']
                p['x'] += p['vx']
            elif p['type'] == 'snow':
                p['angle'] += 0.03
                p['x'] += math.sin(p['angle']) * 0.4
                p['y'] += p['vy']
            p['life'] -= 1
            p['alpha'] = max(0, p['alpha'] - 1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def draw(self, surface: pygame.Surface):
        for p in self.particles:
            if p['type'] == 'rain':
                color = (100, 140, 200)
                pygame.draw.line(surface, color,
                               (int(p['x']), int(p['y'])),
                               (int(p['x']), int(p['y'] + 6)), 1)
            elif p['type'] == 'snow':
                pygame.draw.circle(surface, (255, 255, 255),
                                 (int(p['x']), int(p['y'])), int(p['size']))


class InfoPanel:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 32)

    def draw(self, surface: pygame.Surface, selected_city: Optional[Dict],
             weather_data: Optional[WeatherData], sim_time: datetime, speed: int,
             total_cities: int, storm_count: int):
        pygame.draw.rect(surface, (18, 28, 48), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(surface, (50, 70, 100), (self.x, self.y, self.width, self.height), 2)

        title = self.title_font.render("Weather Dashboard", True, (200, 220, 255))
        surface.blit(title, (self.x + 15, self.y + 12))

        time_str = sim_time.strftime("%Y-%m-%d %H:%M UTC")
        time_text = self.small_font.render(time_str, True, (140, 160, 200))
        surface.blit(time_text, (self.x + 15, self.y + 40))

        speed_str = f"Speed: 1s = {speed // 60} min"
        speed_text = self.small_font.render(speed_str, True, (140, 160, 200))
        surface.blit(speed_text, (self.x + 15, self.y + 58))

        count_text = self.small_font.render(f"Locations: {total_cities} | Storms: {storm_count}", 
                                            True, (140, 160, 200))
        surface.blit(count_text, (self.x + 15, self.y + 76))

        y_offset = self.y + 105
        pygame.draw.line(surface, (40, 60, 90), 
                        (self.x + 15, y_offset - 5), (self.x + self.width - 15, y_offset - 5), 1)

        if selected_city and weather_data:
            self._draw_city_info(surface, selected_city, weather_data, y_offset)
        else:
            self._draw_instructions(surface, y_offset)

        self._draw_legend(surface)
        self._draw_controls(surface)

    def _draw_city_info(self, surface: pygame.Surface, city: Dict, 
                       weather: WeatherData, y_offset: int):
        name_text = self.font.render(city['name'], True, (255, 255, 200))
        surface.blit(name_text, (self.x + 15, y_offset))

        coords = f"Lat: {city['lat']:.4f}° | Lon: {city['lon']:.4f}°"
        coords_text = self.small_font.render(coords, True, (120, 140, 160))
        surface.blit(coords_text, (self.x + 15, y_offset + 25))

        y_offset += 55
        info_items = [
            ("Temperature", f"{weather.temp:.1f}°C", (255, 200, 100)),
            ("Condition", weather.condition.capitalize(),
             WEATHER_GRADIENTS.get(weather.condition, [(200, 200, 200)])[0]),
            ("Humidity", f"{weather.humidity}%", (100, 180, 255)),
            ("Wind", f"{weather.wind_speed:.1f} km/h @ {weather.wind_dir:.0f}°", (100, 255, 150)),
            ("Pressure", f"{weather.pressure:.1f} hPa", (180, 180, 180)),
            ("Visibility", f"{weather.visibility:.1f} km", (180, 180, 180)),
            ("Precipitation", f"{weather.precipitation:.1f} mm", (150, 200, 255)),
            ("Cloud Cover", f"{weather.cloud_cover}%", (160, 160, 180)),
        ]

        for label, value, color in info_items:
            pygame.draw.rect(surface, (25, 35, 55), (self.x + 12, y_offset - 2, self.width - 24, 22))
            label_text = self.small_font.render(f"{label}:", True, (130, 150, 170))
            value_text = self.font.render(value, True, color)
            surface.blit(label_text, (self.x + 15, y_offset))
            surface.blit(value_text, (self.x + 115, y_offset))
            y_offset += 26

    def _draw_instructions(self, surface: pygame.Surface, y_offset: int):
        instructions = [
            "Click on a weather marker",
            "to view detailed data.",
            "",
            "Markers update automatically",
            "from real weather API.",
        ]
        for line in instructions:
            text = self.small_font.render(line, True, (120, 140, 160))
            surface.blit(text, (self.x + 15, y_offset))
            y_offset += 22

    def _draw_legend(self, surface: pygame.Surface):
        y_offset = self.y + 330
        pygame.draw.line(surface, (40, 60, 90),
                        (self.x + 15, y_offset), (self.x + self.width - 15, y_offset), 1)
        y_offset += 8

        title = self.font.render("Weather Types", True, (180, 200, 230))
        surface.blit(title, (self.x + 15, y_offset))
        y_offset += 28

        conditions = [
            ('clear', 'Clear'),
            ('cloudy', 'Cloudy'),
            ('rain', 'Rain'),
            ('storm', 'Storm'),
            ('snow', 'Snow'),
            ('fog', 'Fog'),
            ('wind', 'Wind'),
        ]

        for cond, name in conditions:
            gradient = WEATHER_GRADIENTS[cond]
            for i, color in enumerate(gradient):
                pygame.draw.circle(surface, color, (self.x + 20 + i * 14, y_offset + 6), 5)
            text = self.small_font.render(name, True, (160, 175, 190))
            surface.blit(text, (self.x + 65, y_offset))
            y_offset += 22

    def _draw_controls(self, surface: pygame.Surface):
        y_offset = self.y + 520
        pygame.draw.line(surface, (40, 60, 90),
                        (self.x + 15, y_offset - 5), (self.x + self.width - 15, y_offset - 5), 1)
        y_offset += 3

        title = self.font.render("Controls", True, (180, 200, 230))
        surface.blit(title, (self.x + 15, y_offset))
        y_offset += 25

        controls = [
            ("↑ / ↓", "Adjust speed"),
            ("R", "Refresh data"),
            ("F", "Add front"),
            ("Esc", "Exit"),
        ]

        for key, desc in controls:
            key_text = self.small_font.render(key, True, (150, 180, 200))
            desc_text = self.small_font.render(desc, True, (120, 140, 160))
            surface.blit(key_text, (self.x + 15, y_offset))
            surface.blit(desc_text, (self.x + 60, y_offset))
            y_offset += 20


class WeatherMapGUI:
    def __init__(self, env, lat_min=-85, lat_max=85, lon_min=-180, lon_max=180):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Global Weather Map - Real Data")
        self.clock = pygame.time.Clock()

        self.land_data = LandData()
        self.world_map = WorldMap(MAP_WIDTH, MAP_HEIGHT, MAP_OFFSET_X, MAP_OFFSET_Y, self.land_data, lat_min, lat_max, lon_min, lon_max)
        self.weather_api = WeatherAPI()
        self.front_visualizer = FrontVisualizer()
        self.particle_system = ParticleSystem()

        self.info_panel = InfoPanel(INFO_PANEL_X, MAP_OFFSET_Y, 390, MAP_HEIGHT)

        self.weather_data: Dict[str, WeatherData] = {}
        self.selected_city_key: Optional[str] = None
        self.selected_city: Optional[Dict] = None

        self.running = True
        self.last_update = time.time()
        self.update_interval = 2.0
        self.simulation_speed = 1800
        self.simulated_time = datetime.now(timezone.utc)

        self.animation_time = 0
        self.env = env

        self._initialize_weather()

    def _initialize_weather(self):
        for city in CITY_DATABASE:
            key = f"{city['lat']:.4f}_{city['lon']:.4f}"
            weather = self.weather_api.get_weather(city['lat'], city['lon'], key)
            self.weather_data[key] = weather

    def _get_city_at_position(self, x: int, y: int) -> Optional[Tuple[str, Dict]]:
        for city in CITY_DATABASE:
            cx, cy = self.world_map.lat_lon_to_screen(city['lat'], city['lon'])
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < 18:
                key = f"{city['lat']:.4f}_{city['lon']:.4f}"
                return (key, city)
        return None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    result = self._get_city_at_position(*event.pos)
                    if result:
                        self.selected_city_key, self.selected_city = result
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_UP:
                    self.simulation_speed = min(7200, self.simulation_speed * 2)
                elif event.key == pygame.K_DOWN:
                    self.simulation_speed = max(300, self.simulation_speed // 2)
                elif event.key == pygame.K_r:
                    self._initialize_weather()
                elif event.key == pygame.K_f:
                    self.front_visualizer._add_random_front()

    def update(self):

        theta = random.uniform(0, 2*np.pi) #TO BE REPLACED WITH AGENT'S ACTION

        state, reward, done = self.env.step(theta)

        if done:
            self.env.reset()
        dt = 1.0 / FPS
        self.animation_time += dt

        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            self.simulated_time += timedelta(seconds=self.simulation_speed)

            for city in random.sample(CITY_DATABASE, min(6, len(CITY_DATABASE))):
                key = f"{city['lat']:.4f}_{city['lon']:.4f}"
                if random.random() < 0.6:
                    weather = self.weather_api.get_weather(city['lat'], city['lon'], key)
                    self.weather_data[key] = weather

        self.front_visualizer.update(dt)
        self.particle_system.update()

    def draw(self):
        self.screen.fill(DARK_BLUE)


        self.world_map.draw(self.screen)
        self.world_map.draw_grid(self.screen)

        self.front_visualizer.draw(self.screen, self.world_map, self.animation_time)

        self._draw_weather_markers()

        storm_count = sum(1 for w in self.weather_data.values() if w.condition == 'storm')
        self.info_panel.draw(
            self.screen,
            self.selected_city,
            self.weather_data.get(self.selected_city_key) if self.selected_city_key else None,
            self.simulated_time,
            self.simulation_speed,
            len(CITY_DATABASE),
            storm_count
        )
        ship_lat, ship_lon = self.env.ship_position
        goal_lat, goal_lon = self.env.goal_position

        ship_x, ship_y = self.world_map.lat_lon_to_screen(ship_lat, ship_lon)
        goal_x, goal_y = self.world_map.lat_lon_to_screen(goal_lat, goal_lon)

        pygame.draw.circle(self.screen, (255, 80, 80), (ship_x, ship_y), 6)
        pygame.draw.circle(self.screen, (80, 255, 80), (goal_x, goal_y), 6)

        pygame.display.flip()

    def _draw_weather_markers(self):
        for city in CITY_DATABASE:
            key = f"{city['lat']:.4f}_{city['lon']:.4f}"
            if key not in self.weather_data:
                continue

            weather = self.weather_data[key]
            x, y = self.world_map.lat_lon_to_screen(city['lat'], city['lon'])

            gradient = WEATHER_GRADIENTS.get(weather.condition, WEATHER_GRADIENTS['clear'])
            is_selected = (key == self.selected_city_key)

            base_radius = 14 if is_selected else 10
            pulse = math.sin(self.animation_time * 3) * 0.15 + 0.85

            for i, color in enumerate(reversed(gradient)):
                r = int((base_radius + (len(gradient) - i) * 3) * pulse)
                alpha = int(80 - i * 20)
                alpha_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(alpha_surface, (*color, alpha), (r, r), r)
                self.screen.blit(alpha_surface, (x - r, y - r))

            pygame.draw.circle(self.screen, gradient[0], (x, y), base_radius)

            if is_selected:
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), base_radius + 2, 2)

                angle_offset = self.animation_time * 80
                for i in range(4):
                    angle = math.radians(i * 90 + angle_offset)
                    px = x + math.cos(angle) * (base_radius + 6)
                    py = y + math.sin(angle) * (base_radius + 6)
                    pygame.draw.circle(self.screen, (255, 255, 200), (int(px), int(py)), 2)

            self._draw_mini_icon(x, y, weather.condition, base_radius)

            if weather.condition in ['rain', 'snow']:
                self.particle_system.add_particles(x, y, weather.condition, 0.5)

    def _draw_mini_icon(self, x: int, y: int, condition: str, radius: int):
        size = max(3, radius // 3)

        if condition == 'clear':
            pygame.draw.circle(self.screen, (255, 230, 50), (x, y), size)
            for i in range(6):
                angle = i * 60 + self.animation_time * 40
                rad = math.radians(angle)
                x1 = x + math.cos(rad) * (size + 2)
                y1 = y + math.sin(rad) * (size + 2)
                x2 = x + math.cos(rad) * (size + 4)
                y2 = y + math.sin(rad) * (size + 4)
                pygame.draw.line(self.screen, (255, 230, 50), (x1, y1), (x2, y2), 1)

        elif condition == 'cloudy':
            pygame.draw.circle(self.screen, (170, 175, 180), (x - 2, y - 1), size)
            pygame.draw.circle(self.screen, (180, 185, 190), (x + 2, y + 1), size - 1)

        elif condition == 'rain':
            pygame.draw.circle(self.screen, (140, 150, 160), (x, y - 1), size)
            for i in range(2):
                offset = (i - 0.5) * 4
                pygame.draw.line(self.screen, (70, 110, 180),
                               (int(x + offset), y + size),
                               (int(x + offset - 1), y + size + 3), 1)

        elif condition == 'storm':
            pygame.draw.circle(self.screen, (90, 95, 100), (x, y - 1), size)
            pygame.draw.polygon(self.screen, (255, 255, 120),
                              [(x - 1, y + 2), (x + 1, y + 4), (x - 1, y + 5)])

        elif condition == 'snow':
            for i in range(3):
                pygame.draw.circle(self.screen, (240, 245, 255),
                                 (int(x + (i - 1) * 3), y + (i % 2)), size // 2)

        elif condition == 'fog':
            for i in range(2):
                pygame.draw.line(self.screen, (150, 155, 160),
                               (x - size, y + i * 2 - 1), (x + size, y + i * 2 - 1), 1)

        elif condition == 'wind':
            pygame.draw.arc(self.screen, (100, 180, 150),
                          (x - size, y - size + 1, size * 2 - 1, size * 2 - 1),
                          0, math.pi * 1.3, 1)

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def runGUI(world, env):


    gui = WeatherMapGUI(env,world.lat_min, world.lat_max, world.lon_min, world.lon_max)
    gui.run()

