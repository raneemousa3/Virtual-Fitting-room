import logging
from pathlib import Path
import sys
import json
import time
from typing import Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Add the parent directory to the path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class SizeChartScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.size_charts = {}
        
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Initialize the webdriver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(10)

    def __del__(self):
        """Clean up webdriver"""
        if hasattr(self, 'driver'):
            self.driver.quit()

    def wait_for_element(self, by: By, value: str, timeout: int = 10) -> Optional[webdriver.remote.webelement.WebElement]:
        """Wait for an element to be present on the page"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            return None

    def scrape_zara(self) -> Dict:
        """Scrape size charts from Zara"""
        self.logger.info("Scraping Zara size charts...")
        try:
            # Navigate to Zara's size guide
            self.driver.get("https://www.zara.com/us/en/help/size-guide")
            
            # Wait for the size guide to load
            size_guide = self.wait_for_element(By.CLASS_NAME, "size-guide")
            if not size_guide:
                raise Exception("Size guide not found")
            
            # Initialize size chart dictionary
            size_chart = {}
            
            # Find all measurement tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                # Get the category from the preceding heading
                category = table.find_element(By.XPATH, "./preceding::h2[1]").text.strip().lower()
                
                # Extract headers (sizes)
                headers = []
                for th in table.find_elements(By.TAG_NAME, "th"):
                    size = th.text.strip()
                    if size and size.lower() != "measurements":
                        headers.append(size)
                
                # Extract measurements
                measurements = {}
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header row
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        measurement = cells[0].text.strip().lower()
                        values = [cell.text.strip() for cell in cells[1:]]
                        if measurement and values:
                            measurements[measurement] = dict(zip(headers, values))
                
                if measurements:
                    size_chart[category] = measurements
            
            self.logger.info(f"Successfully scraped Zara size chart with {len(size_chart)} categories")
            return size_chart
            
        except Exception as e:
            self.logger.error(f"Error scraping Zara: {e}")
            return {}

    def scrape_hm(self) -> Dict:
        """Scrape size charts from H&M"""
        self.logger.info("Scraping H&M size charts...")
        try:
            # Navigate to H&M's size guide
            self.driver.get("https://www2.hm.com/en_us/customer-service/sizeguide")
            
            # Wait for the size guide to load
            size_guide = self.wait_for_element(By.CLASS_NAME, "sizeguide-content")
            if not size_guide:
                raise Exception("Size guide not found")
            
            # Initialize size chart dictionary
            size_chart = {}
            
            # Find all measurement tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                # Get the category from the preceding heading
                category = table.find_element(By.XPATH, "./preceding::h3[1]").text.strip().lower()
                
                # Extract headers (sizes)
                headers = []
                for th in table.find_elements(By.TAG_NAME, "th"):
                    size = th.text.strip()
                    if size and size.lower() != "measurements":
                        headers.append(size)
                
                # Extract measurements
                measurements = {}
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header row
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        measurement = cells[0].text.strip().lower()
                        values = [cell.text.strip() for cell in cells[1:]]
                        if measurement and values:
                            measurements[measurement] = dict(zip(headers, values))
                
                if measurements:
                    size_chart[category] = measurements
            
            self.logger.info(f"Successfully scraped H&M size chart with {len(size_chart)} categories")
            return size_chart
            
        except Exception as e:
            self.logger.error(f"Error scraping H&M: {e}")
            return {}

    def scrape_uniqlo(self) -> Dict:
        """Scrape size charts from Uniqlo"""
        self.logger.info("Scraping Uniqlo size charts...")
        try:
            # Navigate to Uniqlo's size guide
            self.driver.get("https://www.uniqlo.com/us/en/size-chart")
            
            # Wait for the size guide to load
            size_guide = self.wait_for_element(By.CLASS_NAME, "size-chart")
            if not size_guide:
                raise Exception("Size guide not found")
            
            # Initialize size chart dictionary
            size_chart = {}
            
            # Find all measurement tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                # Get the category from the preceding heading
                category = table.find_element(By.XPATH, "./preceding::h2[1]").text.strip().lower()
                
                # Extract headers (sizes)
                headers = []
                for th in table.find_elements(By.TAG_NAME, "th"):
                    size = th.text.strip()
                    if size and size.lower() != "measurements":
                        headers.append(size)
                
                # Extract measurements
                measurements = {}
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header row
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        measurement = cells[0].text.strip().lower()
                        values = [cell.text.strip() for cell in cells[1:]]
                        if measurement and values:
                            measurements[measurement] = dict(zip(headers, values))
                
                if measurements:
                    size_chart[category] = measurements
            
            self.logger.info(f"Successfully scraped Uniqlo size chart with {len(size_chart)} categories")
            return size_chart
            
        except Exception as e:
            self.logger.error(f"Error scraping Uniqlo: {e}")
            return {}

    def save_size_charts(self):
        """Save the scraped size charts to a JSON file"""
        output_dir = Path("data/size_charts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "size_charts.json"
        with open(output_file, 'w') as f:
            json.dump(self.size_charts, f, indent=4)
        
        self.logger.info(f"Size charts saved to {output_file}")

def main():
    """Main function to scrape size charts"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        scraper = SizeChartScraper()
        
        # Scrape each retailer
        scraper.size_charts['zara'] = scraper.scrape_zara()
        time.sleep(2)  # Be nice to the servers
        
        scraper.size_charts['hm'] = scraper.scrape_hm()
        time.sleep(2)
        
        scraper.size_charts['uniqlo'] = scraper.scrape_uniqlo()
        
        # Save the results
        scraper.save_size_charts()
        
        logger.info("Size chart scraping completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        sys.exit(1)
    finally:
        if 'scraper' in locals():
            del scraper  # This will trigger the __del__ method to clean up the webdriver

if __name__ == "__main__":
    main() 