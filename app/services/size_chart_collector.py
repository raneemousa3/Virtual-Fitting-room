import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import Dict, List, Optional
import logging
from pathlib import Path

class SizeChartCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/size_charts")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard measurement keys
        self.measurement_keys = {
            'chest': ['chest', 'bust', 'chest/bust'],
            'waist': ['waist', 'natural waist'],
            'hips': ['hips', 'hip', 'seat'],
            'height': ['height', 'tall'],
            'shoulder': ['shoulder', 'shoulder width'],
            'sleeve': ['sleeve', 'sleeve length'],
            'inseam': ['inseam', 'inside leg']
        }

    def collect_hm_size_chart(self) -> Dict:
        """Collect H&M size chart data"""
        try:
            # H&M size chart URL
            url = "https://www2.hm.com/en_gb/customer-service/size-guide.html"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize size data dictionary
            size_data = {}
            
            # Find the size chart table
            size_table = soup.find('table', {'class': 'size-chart'})
            if not size_table:
                self.logger.warning("H&M size chart table not found")
                return {}
            
            # Extract headers (sizes)
            headers = []
            for th in size_table.find_all('th'):
                size = th.text.strip()
                if size and size != 'Measurements':
                    headers.append(size)
            
            # Extract measurements
            for row in size_table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) >= 2:
                    measurement = cells[0].text.strip().lower()
                    standard_key = self._find_standard_key(measurement)
                    
                    if standard_key:
                        for i, size in enumerate(headers):
                            if i + 1 < len(cells):
                                value = cells[i + 1].text.strip()
                                if size not in size_data:
                                    size_data[size] = {}
                                size_data[size][standard_key] = self._standardize_value(value)
            
            return size_data
            
        except Exception as e:
            self.logger.error(f"Error collecting H&M size chart: {e}")
            return {}

    def collect_zara_size_chart(self) -> Dict:
        """Collect Zara size chart data"""
        try:
            # Zara size chart URL
            url = "https://www.zara.com/uk/en/size-guide-l1452.html"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize size data dictionary
            size_data = {}
            
            # Find the size chart section
            size_section = soup.find('div', {'class': 'size-guide'})
            if not size_section:
                self.logger.warning("Zara size guide section not found")
                return {}
            
            # Extract size tables
            size_tables = size_section.find_all('table')
            for table in size_tables:
                # Get category (e.g., "Women", "Men")
                category = table.find_previous('h2')
                category = category.text.strip() if category else "Unknown"
                
                # Extract headers (sizes)
                headers = []
                header_row = table.find('tr')
                if header_row:
                    for th in header_row.find_all('th'):
                        size = th.text.strip()
                        if size and size != 'Measurements':
                            headers.append(size)
                
                # Extract measurements
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        measurement = cells[0].text.strip().lower()
                        standard_key = self._find_standard_key(measurement)
                        
                        if standard_key:
                            for i, size in enumerate(headers):
                                if i + 1 < len(cells):
                                    value = cells[i + 1].text.strip()
                                    size_key = f"{category}_{size}"
                                    if size_key not in size_data:
                                        size_data[size_key] = {}
                                    size_data[size_key][standard_key] = self._standardize_value(value)
            
            return size_data
            
        except Exception as e:
            self.logger.error(f"Error collecting Zara size chart: {e}")
            return {}

    def collect_uniqlo_size_chart(self) -> Dict:
        """Collect Uniqlo size chart data"""
        try:
            # Uniqlo size chart URL
            url = "https://www.uniqlo.com/uk/en/size-guide"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize size data dictionary
            size_data = {}
            
            # Find all size guide sections
            size_sections = soup.find_all('div', {'class': 'size-guide-section'})
            for section in size_sections:
                # Get category (e.g., "Men", "Women", "Kids")
                category = section.find('h2')
                category = category.text.strip() if category else "Unknown"
                
                # Find size tables
                size_tables = section.find_all('table')
                for table in size_tables:
                    # Extract headers (sizes)
                    headers = []
                    header_row = table.find('tr')
                    if header_row:
                        for th in header_row.find_all('th'):
                            size = th.text.strip()
                            if size and size != 'Size':
                                headers.append(size)
                    
                    # Extract measurements
                    for row in table.find_all('tr')[1:]:  # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            measurement = cells[0].text.strip().lower()
                            standard_key = self._find_standard_key(measurement)
                            
                            if standard_key:
                                for i, size in enumerate(headers):
                                    if i + 1 < len(cells):
                                        value = cells[i + 1].text.strip()
                                        size_key = f"{category}_{size}"
                                        if size_key not in size_data:
                                            size_data[size_key] = {}
                                        size_data[size_key][standard_key] = self._standardize_value(value)
            
            return size_data
            
        except Exception as e:
            self.logger.error(f"Error collecting Uniqlo size chart: {e}")
            return {}

    def standardize_measurements(self, raw_data: Dict) -> Dict:
        """Standardize measurements from different sources"""
        standardized_data = {}
        
        for brand, measurements in raw_data.items():
            standardized_data[brand] = {}
            for size, data in measurements.items():
                standardized_data[brand][size] = {}
                for key, value in data.items():
                    # Find matching standard key
                    standard_key = self._find_standard_key(key)
                    if standard_key:
                        standardized_data[brand][size][standard_key] = self._standardize_value(value)
        
        return standardized_data

    def _find_standard_key(self, key: str) -> Optional[str]:
        """Find standard measurement key for given key"""
        key = key.lower().strip()
        for standard_key, variations in self.measurement_keys.items():
            if key in variations:
                return standard_key
        return None

    def _standardize_value(self, value: str) -> float:
        """Convert measurement value to standard format (cm)"""
        try:
            # Remove any non-numeric characters except decimal point
            value = ''.join(c for c in value if c.isdigit() or c == '.')
            return float(value)
        except ValueError:
            return 0.0

    def save_size_chart_data(self, data: Dict, filename: str):
        """Save size chart data to JSON file"""
        file_path = self.data_dir / filename
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Saved size chart data to {file_path}")

    def collect_all_size_charts(self):
        """Collect size charts from all sources"""
        all_data = {}
        
        # Collect from each source
        all_data['hm'] = self.collect_hm_size_chart()
        all_data['zara'] = self.collect_zara_size_chart()
        all_data['uniqlo'] = self.collect_uniqlo_size_chart()
        
        # Standardize measurements
        standardized_data = self.standardize_measurements(all_data)
        
        # Save to file
        self.save_size_chart_data(standardized_data, 'size_charts.json')
        
        return standardized_data

    def load_size_chart_data(self, filename: str = 'size_charts.json') -> Dict:
        """Load size chart data from JSON file"""
        file_path = self.data_dir / filename
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Size chart data file not found: {file_path}")
            return {} 