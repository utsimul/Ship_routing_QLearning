

import pygame
import math
import random
import time
import numpy as np
import shapefile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
from weather.weather_simulator import WeatherSimulator

pygame.init()

SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 950
FPS = 60

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

WEATHER_COLORMAP = [
    (25, 50, 100),
    (40, 80, 150),
    (60, 120, 180),
    (80, 150, 180),
    (100, 170, 140),
    (140, 190, 100),
    (190, 210, 80),
    (220, 190, 90),
    (230, 160, 100),
    (210, 120, 130),
    (180, 90, 160),
    (140, 60, 180),
]

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
    {'name': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194, 'country': 'US'},
    {'name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321, 'country': 'US'},
    {'name': 'Denver', 'lat': 39.7392, 'lon': -104.9903, 'country': 'US'},
    {'name': 'Boston', 'lat': 42.3601, 'lon': -71.0589, 'country': 'US'},
    {'name': 'Nashville', 'lat': 36.1627, 'lon': -86.7816, 'country': 'US'},
    {'name': 'Detroit', 'lat': 42.3314, 'lon': -83.0458, 'country': 'US'},
    {'name': 'Portland', 'lat': 45.5152, 'lon': -122.6784, 'country': 'US'},
    {'name': 'Las Vegas', 'lat': 36.1699, 'lon': -115.1398, 'country': 'US'},
    {'name': 'Toronto', 'lat': 43.6532, 'lon': -79.3832, 'country': 'CA'},
    {'name': 'Vancouver', 'lat': 49.2827, 'lon': -123.1207, 'country': 'CA'},
    {'name': 'Montreal', 'lat': 45.5017, 'lon': -73.5673, 'country': 'CA'},
    {'name': 'Calgary', 'lat': 51.0447, 'lon': -114.0719, 'country': 'CA'},
    {'name': 'Mexico City', 'lat': 19.4326, 'lon': -99.1332, 'country': 'MX'},
    {'name': 'Guadalajara', 'lat': 20.6597, 'lon': -103.3496, 'country': 'MX'},
    {'name': 'Sao Paulo', 'lat': -23.5505, 'lon': -46.6333, 'country': 'BR'},
    {'name': 'Rio de Janeiro', 'lat': -22.9068, 'lon': -43.1729, 'country': 'BR'},
    {'name': 'Buenos Aires', 'lat': -34.6037, 'lon': -58.3816, 'country': 'AR'},
    {'name': 'Lima', 'lat': -12.0464, 'lon': -77.0428, 'country': 'PE'},
    {'name': 'Bogota', 'lat': 4.7110, 'lon': -74.0721, 'country': 'CO'},
    {'name': 'Santiago', 'lat': -33.4489, 'lon': -70.6693, 'country': 'CL'},
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
    {'name': 'Stockholm', 'lat': 59.3293, 'lon': 18.0686, 'country': 'SE'},
    {'name': 'Oslo', 'lat': 59.9139, 'lon': 10.7522, 'country': 'NO'},
    {'name': 'Copenhagen', 'lat': 55.6761, 'lon': 12.5683, 'country': 'DK'},
    {'name': 'Helsinki', 'lat': 60.1699, 'lon': 24.9384, 'country': 'FI'},
    {'name': 'Dublin', 'lat': 53.3498, 'lon': -6.2603, 'country': 'IE'},
    {'name': 'Moscow', 'lat': 55.7558, 'lon': 37.6173, 'country': 'RU'},
    {'name': 'Istanbul', 'lat': 41.0082, 'lon': 28.9784, 'country': 'TR'},
    {'name': 'Cairo', 'lat': 30.0444, 'lon': 31.2357, 'country': 'EG'},
    {'name': 'Cape Town', 'lat': -33.9249, 'lon': 18.4241, 'country': 'ZA'},
    {'name': 'Johannesburg', 'lat': -26.2041, 'lon': 28.0473, 'country': 'ZA'},
    {'name': 'Lagos', 'lat': 6.5244, 'lon': 3.3792, 'country': 'NG'},
    {'name': 'Dubai', 'lat': 25.2048, 'lon': 55.2708, 'country': 'AE'},
    {'name': 'Delhi', 'lat': 28.7041, 'lon': 77.1025, 'country': 'IN'},
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'country': 'IN'},
    {'name': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639, 'country': 'IN'},
    {'name': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946, 'country': 'IN'},
    {'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011, 'country': 'PK'},
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
    {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198, 'country': 'SG'},
    {'name': 'Bangkok', 'lat': 13.7563, 'lon': 100.5018, 'country': 'TH'},
    {'name': 'Kuala Lumpur', 'lat': 3.1390, 'lon': 101.6869, 'country': 'MY'},
    {'name': 'Jakarta', 'lat': -6.2088, 'lon': 106.8456, 'country': 'ID'},
    {'name': 'Manila', 'lat': 14.5995, 'lon': 120.9842, 'country': 'PH'},
    {'name': 'Hanoi', 'lat': 21.0285, 'lon': 105.8542, 'country': 'VN'},
    {'name': 'Sydney', 'lat': -33.8688, 'lon': 151.2093, 'country': 'AU'},
    {'name': 'Melbourne', 'lat': -37.8136, 'lon': 144.9631, 'country': 'AU'},
    {'name': 'Brisbane', 'lat': -27.4698, 'lon': 153.0251, 'country': 'AU'},
    {'name': 'Perth', 'lat': -31.9505, 'lon': 115.8605, 'country': 'AU'},
    {'name': 'Auckland', 'lat': -36.8509, 'lon': 174.7645, 'country': 'NZ'},
]


class World:
    def __init__(self, width: int, height: int, offset_x: int, offset_y: int):
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.min_lat = -85
        self.max_lat = 85
        self.min_lon = -180
        self.max_lon = 180
        # self.weather = np.zeros((height, width))
        self.wind_u = np.zeros((height, width))
        self.wind_v = np.zeros((height, width))
        self.storm_data = []

    def lat_lon_to_screen(self, lat: float, lon: float) -> Tuple[int, int]:
        x = self.offset_x + int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.width)
        y = self.offset_y + int((self.max_lat - lat) / (self.max_lat - self.min_lat) * self.height)
        return (x, y)

    def screen_to_lat(self, y: int) -> float:
        return self.max_lat - (y - self.offset_y) / self.height * (self.max_lat - self.min_lat)

    def screen_to_lon(self, x: int) -> float:
        return self.min_lon + (x - self.offset_x) / self.width * (self.max_lon - self.min_lon)

    def screen_to_lat_lon(self, x: int, y: int) -> Tuple[float, float]:
        return (self.screen_to_lat(y), self.screen_to_lon(x))

    def lat_lon_to_grid(self, lat: float, lon: float):

        j = int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.width)
        i = int((self.max_lat - lat) / (self.max_lat - self.min_lat) * self.height)

        i = max(0, min(self.height - 1, i))
        j = max(0, min(self.width - 1, j))

        return i, j


class LandData:
    def __init__(self):
        self.polygons: List[List[Tuple[float, float]]] = []
        self._load_shapefile()

    def _load_shapefile(self):
        try:
            sf = shapefile.Reader("ne_50m_land/ne_50m_land.shp")
            for shape_record in sf.shapeRecords():
                shape = shape_record.shape
                if shape.shapeType in (shapefile.POLYGON, shapefile.POLYLINE):
                    parts = list(shape.parts) + [len(shape.points)]
                    for i in range(len(parts) - 1):
                        start = parts[i]
                        end = parts[i + 1]
                        points = [(float(shape.points[j][0]), float(shape.points[j][1])) 
                                  for j in range(start, end)]
                        if len(points) >= 3:
                            self.polygons.append(points)
            print(f"Loaded {len(self.polygons)} land polygons from shapefile")
        except Exception as e:
            print(f"Could not load shapefile: {e}")
            self._load_fallback_data()

    def _load_fallback_data(self):
        print("Using fallback land data")
        continents = [
            [(-170.0, 65.0), (-140.0, 72.0), (-100.0, 72.0), (-80.0, 65.0), (-75.0, 45.0), (-55.0, 45.0), (-60.0, 25.0), (-85.0, 10.0), (-120.0, 25.0), (-125.0, 35.0), (-115.0, 32.0), (-105.0, 25.0), (-95.0, 20.0), (-85.0, 20.0), (-82.0, 25.0), (-80.0, 30.0), (-75.0, 35.0), (-70.0, 42.0), (-68.0, 45.0), (-55.0, 48.0), (-68.0, 55.0), (-80.0, 50.0), (-95.0, 50.0), (-105.0, 45.0), (-115.0, 35.0), (-125.0, 40.0), (-135.0, 55.0), (-150.0, 60.0), (-170.0, 65.0)],
            [(-82.0, 10.0), (-75.0, 5.0), (-55.0, -5.0), (-40.0, -5.0), (-35.0, -10.0), (-35.0, -25.0), (-55.0, -55.0), (-70.0, -55.0), (-75.0, -45.0), (-75.0, -30.0), (-80.0, -5.0), (-82.0, 10.0)],
            [(-10.0, 35.0), (5.0, 45.0), (10.0, 45.0), (30.0, 45.0), (30.0, 55.0), (25.0, 60.0), (30.0, 70.0), (60.0, 70.0), (70.0, 55.0), (60.0, 35.0), (50.0, 35.0), (40.0, 35.0), (35.0, 40.0), (25.0, 35.0), (15.0, 40.0), (5.0, 42.0), (-10.0, 35.0)],
            [(-20.0, 35.0), (-5.0, 42.0), (10.0, 45.0), (15.0, 42.0), (25.0, 35.0), (35.0, 35.0), (45.0, 30.0), (50.0, 15.0), (55.0, 15.0), (60.0, 20.0), (50.0, 15.0), (40.0, 5.0), (35.0, -5.0), (35.0, -15.0), (30.0, -35.0), (25.0, -35.0), (15.0, -30.0), (10.0, -5.0), (-10.0, 5.0), (-15.0, 15.0), (-20.0, 35.0)],
            [(60.0, 35.0), (75.0, 45.0), (90.0, 50.0), (105.0, 55.0), (120.0, 50.0), (135.0, 50.0), (145.0, 45.0), (155.0, 50.0), (165.0, 60.0), (175.0, 65.0), (170.0, 65.0), (160.0, 55.0), (140.0, 50.0), (130.0, 50.0), (120.0, 45.0), (110.0, 35.0), (100.0, 30.0), (90.0, 25.0), (80.0, 15.0), (80.0, 10.0), (75.0, 5.0), (100.0, 10.0), (105.0, 5.0), (120.0, -5.0), (130.0, -5.0), (135.0, -5.0), (140.0, -15.0), (145.0, -35.0), (165.0, -40.0), (175.0, -40.0), (170.0, -35.0), (160.0, -25.0), (150.0, -15.0), (140.0, -5.0), (130.0, 5.0), (120.0, 15.0), (110.0, 25.0), (100.0, 25.0), (90.0, 25.0), (80.0, 25.0), (70.0, 20.0), (60.0, 30.0), (60.0, 35.0)],
            [(140.0, -15.0), (150.0, -10.0), (155.0, -15.0), (155.0, -25.0), (150.0, -40.0), (145.0, -40.0), (135.0, -35.0), (135.0, -25.0), (140.0, -15.0)],
        ]
        self.polygons = continents


class WeatherRenderer:
    def __init__(self):
        self.colormap = WEATHER_COLORMAP

    def render_heatmap(self, field: np.ndarray) -> pygame.Surface:
        height, width = field.shape
        surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Fixed normalization (important for RL consistency)
        max_wind = 20.0
        normalized = np.clip(field / max_wind, 0, 1)

        step = 3
        for i in range(0, height, step):
            for j in range(0, width, step):
                value = np.mean(normalized[i:i+step, j:j+step])
                color = self._value_to_color(value)
                
                rect_surface = pygame.Surface((step, step), pygame.SRCALPHA)
                alpha = int(60 + value * 80)
                rect_surface.fill((*color, alpha))
                surface.blit(rect_surface, (j, i))

        return surface

    def _value_to_color(self, value: float) -> Tuple[int, int, int]:
        value = np.clip(value, 0, 1)
        idx = int(value * (len(self.colormap) - 1))
        idx = min(idx, len(self.colormap) - 1)
        return self.colormap[idx]

    def render_storm_overlay(self, storms: List[Dict], world: World, 
                            time_val: float) -> pygame.Surface:
        surface = pygame.Surface((world.width, world.height), pygame.SRCALPHA)
        
        for storm in storms:
            x, y = world.lat_lon_to_screen(storm['lat'], storm['lon'])
            x -= world.offset_x
            y -= world.offset_y
            
            intensity = storm.get('intensity', 0.5)
            radius = int(storm.get('radius', 5) * 12)
            
            pulse = math.sin(time_val * 2) * 0.15 + 0.85
            
            for i in range(5):
                r = int(radius * (1 - i * 0.15) * pulse)
                if r > 0:
                    alpha = int((90 - i * 15) * intensity)
                    color = (120, 50, 160, alpha)
                    circle_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(circle_surface, (120, 50, 160), (r, r), r)
                    surface.blit(circle_surface, (int(x - r), int(y - r)))
                    
            for i in range(8):
                angle = math.radians(i * 45 + time_val * 60)
                inner_r = radius * 0.3
                outer_r = radius * 0.7
                x1 = int(x + math.cos(angle) * inner_r)
                y1 = int(y + math.sin(angle) * inner_r)
                x2 = int(x + math.cos(angle) * outer_r)
                y2 = int(y + math.sin(angle) * outer_r)
                pygame.draw.line(surface, (160, 80, 200), (x1, y1), (x2, y2), 2)

        return surface


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
             weather_info: Optional[Dict], sim_time: datetime, speed: int,
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

        if selected_city and weather_info:
            self._draw_city_info(surface, selected_city, weather_info, y_offset)
        else:
            self._draw_instructions(surface, y_offset)

        self._draw_legend(surface)
        self._draw_controls(surface)
    
    def classify_wind(speed):
        if speed < 3: return "calm"
        elif speed < 8: return "breeze"
        elif speed < 15: return "windy"
        else: return "storm"

    def _draw_city_info(self, surface: pygame.Surface, city: Dict, 
                   weather_info: Dict, y_offset: int):

        name_text = self.font.render(city['name'], True, (255, 255, 200))
        surface.blit(name_text, (self.x + 15, y_offset))

        coords = f"Lat: {city['lat']:.4f}° | Lon: {city['lon']:.4f}°"
        coords_text = self.small_font.render(coords, True, (120, 140, 160))
        surface.blit(coords_text, (self.x + 15, y_offset + 25))

        y_offset += 55

        speed = weather_info['speed']
        direction = weather_info['direction']

        # --- Wind Speed ---
        speed_text = self.font.render(f"Wind: {speed:.2f} m/s", True, (180, 200, 220))
        surface.blit(speed_text, (self.x + 15, y_offset))
        y_offset += 25

        # --- Direction ---
        dir_text = self.font.render(f"Direction: {direction:.1f}°", True, (180, 200, 220))
        surface.blit(dir_text, (self.x + 15, y_offset))
        y_offset += 30

        # --- Derived Label (optional but useful) ---
        if speed < 3:
            label = "Calm"
        elif speed < 8:
            label = "Breeze"
        elif speed < 15:
            label = "Windy"
        else:
            label = "Storm"

        label_text = self.font.render(label, True, (200, 220, 255))
        surface.blit(label_text, (self.x + 15, y_offset))
        y_offset += 30

        # --- Speed Bar ---
        pygame.draw.rect(surface, (40, 50, 70), (self.x + 15, y_offset, self.width - 40, 15))

        norm = min(speed / 20.0, 1.0)
        bar_width = int((self.width - 40) * norm)

        color = self.weather_renderer._value_to_color(norm)

        pygame.draw.rect(surface, color, (self.x + 15, y_offset, bar_width, 15))
        pygame.draw.rect(surface, (80, 100, 130), (self.x + 15, y_offset, self.width - 40, 15), 1)

    def _draw_instructions(self, surface: pygame.Surface, y_offset: int):
        instructions = [
            "Click on a weather marker",
            "to view detailed data.",
            "",
            "Weather heatmap shows",
            "continuous simulation data.",
        ]
        for line in instructions:
            text = self.small_font.render(line, True, (120, 140, 160))
            surface.blit(text, (self.x + 15, y_offset))
            y_offset += 22

    def _draw_legend(self, surface: pygame.Surface):
        y_offset = self.y + 290
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
        ]

        for cond, name in conditions:
            gradient = WEATHER_GRADIENTS[cond]
            for i, color in enumerate(gradient):
                pygame.draw.circle(surface, color, (self.x + 20 + i * 14, y_offset + 6), 5)
            text = self.small_font.render(name, True, (160, 175, 190))
            surface.blit(text, (self.x + 65, y_offset))
            y_offset += 22

        y_offset += 10
        scale_title = self.small_font.render("Intensity Scale", True, (180, 200, 230))
        surface.blit(scale_title, (self.x + 15, y_offset))
        y_offset += 22
        
        for i, color in enumerate(WEATHER_COLORMAP):
            pygame.draw.rect(surface, color, (self.x + 20 + i * 28, y_offset, 26, 18))
        
        low_text = self.small_font.render("0", True, (120, 140, 160))
        high_text = self.small_font.render("1", True, (120, 140, 160))
        surface.blit(low_text, (self.x + 15, y_offset + 20))
        surface.blit(high_text, (self.x + self.width - 25, y_offset + 20))

    def _draw_controls(self, surface: pygame.Surface):
        y_offset = self.y + 470
        pygame.draw.line(surface, (40, 60, 90),
                        (self.x + 15, y_offset - 5), (self.x + self.width - 15, y_offset - 5), 1)
        y_offset += 3

        title = self.font.render("Controls", True, (180, 200, 230))
        surface.blit(title, (self.x + 15, y_offset))
        y_offset += 25

        controls = [
            ("↑ / ↓", "Adjust speed"),
            ("F", "Add storm"),
            ("Esc", "Exit"),
        ]

        for key, desc in controls:
            key_text = self.small_font.render(key, True, (150, 180, 200))
            desc_text = self.small_font.render(desc, True, (120, 140, 160))
            surface.blit(key_text, (self.x + 15, y_offset))
            surface.blit(desc_text, (self.x + 60, y_offset))
            y_offset += 20


class WeatherMapGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Global Weather Simulation")
        self.clock = pygame.time.Clock()

        self.world = World(MAP_WIDTH, MAP_HEIGHT, MAP_OFFSET_X, MAP_OFFSET_Y)
        self.weather_simulator = WeatherSimulator(self.world)
        self.weather_renderer = WeatherRenderer()
        self.land_data = LandData()
        self.scaled_polygons = self._scale_land_polygons()

        self.info_panel = InfoPanel(INFO_PANEL_X, MAP_OFFSET_Y, 390, MAP_HEIGHT)

        self.selected_city: Optional[Dict] = None
        self.selected_weather_info: Optional[Dict] = None

        self.running = True
        self.last_update = time.time()
        self.update_interval = 0.05
        self.simulation_speed = 1800
        self.simulated_time = datetime.now(timezone.utc)

        self.animation_time = 0.0

    def _scale_land_polygons(self) -> List[List[Tuple[int, int]]]:
        scaled = []
        for polygon in self.land_data.polygons:
            poly_scaled = []
            for lon, lat in polygon:
                x, y = self.world.lat_lon_to_screen(lat, lon)
                poly_scaled.append((x, y))
            if len(poly_scaled) >= 3:
                scaled.append(poly_scaled)
        return scaled

    def _get_city_at_position(self, x: int, y: int) -> Optional[Dict]:
        for city in CITY_DATABASE:
            cx, cy = self.world.lat_lon_to_screen(city['lat'], city['lon'])
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < 18:
                return city
        return None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    city = self._get_city_at_position(*event.pos)
                    if city:
                        self.selected_city = city
                        self.selected_weather_info = self.weather_simulator.get_weather_at_lat_lon(
                            city['lat'], city['lon']
                        )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_UP:
                    self.simulation_speed = min(7200, self.simulation_speed * 2)
                elif event.key == pygame.K_DOWN:
                    self.simulation_speed = max(300, self.simulation_speed // 2)
                elif event.key == pygame.K_f:
                    self.weather_simulator._spawn_storm()

    def update(self):
        dt = 1.0 / FPS
        self.animation_time += dt

        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            self.simulated_time += timedelta(seconds=self.simulation_speed)
            self.weather_simulator.update()
            
            if self.selected_city:
                self.selected_weather_info = self.weather_simulator.get_weather_at_lat_lon(
                    self.selected_city['lat'], self.selected_city['lon']
                )

    def draw(self):
        self.screen.fill(DARK_BLUE)

        self.screen.fill(OCEAN_COLOR, (MAP_OFFSET_X - 5, MAP_OFFSET_Y - 5,
                                       MAP_WIDTH + 10, MAP_HEIGHT + 10))

        wind_speed = np.sqrt(self.world.wind_u**2 + self.world.wind_v**2)
        heatmap_surface = self.weather_renderer.render_heatmap(wind_speed) 
        self.screen.blit(heatmap_surface, (MAP_OFFSET_X, MAP_OFFSET_Y))

        for polygon in self.scaled_polygons:
            if len(polygon) >= 3:
                pygame.draw.polygon(self.screen, LAND_COLOR, polygon)
                pygame.draw.polygon(self.screen, COAST_COLOR, polygon, 1)

        storm_overlay = self.weather_renderer.render_storm_overlay(
            self.world.storm_data, self.world, self.animation_time
        )
        self.screen.blit(storm_overlay, (MAP_OFFSET_X, MAP_OFFSET_Y))

        self._draw_weather_markers()

        self._draw_grid()

        storm_count = len(self.world.storm_data)
        self.info_panel.draw(
            self.screen,
            self.selected_city,
            self.selected_weather_info,
            self.simulated_time,
            self.simulation_speed,
            len(CITY_DATABASE),
            storm_count
        )

        pygame.display.flip()

    def _draw_grid(self):
        grid_color = (40, 50, 70)
        label_color = (80, 100, 130)
        font = pygame.font.Font(None, 18)

        for lat in range(-80, 90, 20):
            x1, y1 = self.world.lat_lon_to_screen(lat, -180)
            x2, y2 = self.world.lat_lon_to_screen(lat, 180)
            pygame.draw.line(self.screen, grid_color, (x1, y1), (x2, y2), 1)
            label = font.render(f"{lat}°", True, label_color)
            self.screen.blit(label, (MAP_OFFSET_X - 30, y1 - 6))

        for lon in range(-180, 190, 30):
            x1, y1 = self.world.lat_lon_to_screen(-85, lon)
            x2, y2 = self.world.lat_lon_to_screen(85, lon)
            pygame.draw.line(self.screen, grid_color, (x1, y1), (x2, y2), 1)
            if lon != -180:
                label = font.render(f"{lon}°", True, label_color)
                self.screen.blit(label, (x1 - 10, MAP_OFFSET_Y + MAP_HEIGHT + 5))

    def _draw_weather_markers(self):
        for city in CITY_DATABASE:
            x, y = self.world.lat_lon_to_screen(city['lat'], city['lon'])
            
            weather_info = self.weather_simulator.get_weather_at_lat_lon(
                city['lat'], city['lon']
            )

            speed = weather_info['speed']

            # Normalize speed
            norm = min(speed / 20.0, 1.0)

            # Get color from heatmap scale
            color = self.weather_renderer._value_to_color(norm)

            is_selected = (city == self.selected_city)
            base_radius = 14 if is_selected else 10

            pulse = math.sin(self.animation_time * 3) * 0.12 + 0.88

            # Glow effect
            for i in range(3):
                r = int((base_radius + (3 - i) * 3) * pulse)
                alpha = int(70 - i * 20)

                glow_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*color, alpha), (r, r), r)
                self.screen.blit(glow_surface, (x - r, y - r))

            # Main circle
            pygame.draw.circle(self.screen, color, (x, y), base_radius)

            # Selection highlight
            if is_selected:
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), base_radius + 2, 2)

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

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main():
    gui = WeatherMapGUI()
    gui.run()


if __name__ == "__main__":
    main()