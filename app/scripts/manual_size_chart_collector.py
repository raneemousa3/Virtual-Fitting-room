import logging
from pathlib import Path
import sys
import json
import webbrowser
import time
from typing import Dict, List, Optional

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

class SizeChartCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.size_charts = {}
        self.measurements = [
            "chest",
            "waist",
            "hips",
            "shoulder",
            "sleeve",
            "inseam"
        ]
        self.sizes = ["XS", "S", "M", "L", "XL", "XXL"]
        
    def open_retailer_website(self, retailer: str):
        """Open the retailer's size guide in the default browser"""
        urls = {
            "zara": "https://www.zara.com/us/en/help/size-guide",
            "hm": "https://www2.hm.com/en_us/customer-service/size-guide.html",
            "uniqlo": "https://www.uniqlo.com/us/en/size-chart"
        }
        
        if retailer in urls:
            self.logger.info(f"Opening {retailer.upper()} size guide in browser...")
            webbrowser.open(urls[retailer])
            return True
        else:
            self.logger.error(f"Unknown retailer: {retailer}")
            return False
    
    def collect_measurements(self, retailer: str) -> Dict:
        """Collect measurements for a retailer"""
        self.logger.info(f"Collecting measurements for {retailer.upper()}...")
        
        # Open the retailer's website
        if not self.open_retailer_website(retailer):
            return {}
        
        # Give user time to view the website
        print(f"\nPlease view the {retailer.upper()} size guide in your browser.")
        print("Press Enter when you're ready to input the measurements...")
        input()
        
        # Collect measurements for each size
        measurements_data = {}
        for size in self.sizes:
            print(f"\nEntering measurements for size {size}:")
            size_data = {}
            
            for measurement in self.measurements:
                while True:
                    try:
                        value = input(f"{measurement.capitalize()} (cm): ")
                        if value.strip() == "":
                            print("Skipping this measurement...")
                            break
                        
                        # Convert to float and validate
                        value_float = float(value)
                        if value_float <= 0:
                            print("Measurement must be positive. Please try again.")
                            continue
                        
                        size_data[measurement] = value_float
                        break
                    except ValueError:
                        print("Please enter a valid number.")
            
            if size_data:
                measurements_data[size] = size_data
        
        return measurements_data
    
    def collect_all_retailers(self):
        """Collect size charts from all retailers"""
        retailers = ["zara", "hm", "uniqlo"]
        
        for retailer in retailers:
            print(f"\n{'='*50}")
            print(f"COLLECTING DATA FOR {retailer.upper()}")
            print(f"{'='*50}")
            
            measurements = self.collect_measurements(retailer)
            if measurements:
                self.size_charts[retailer] = measurements
                print(f"\nSuccessfully collected {len(measurements)} sizes for {retailer.upper()}")
            else:
                print(f"\nNo measurements collected for {retailer.upper()}")
            
            # Ask if user wants to continue
            if retailer != retailers[-1]:
                print("\nPress Enter to continue to the next retailer, or 'q' to quit...")
                if input().lower() == 'q':
                    break
    
    def save_size_charts(self):
        """Save the collected size charts to a JSON file"""
        if not self.size_charts:
            self.logger.warning("No size charts to save")
            return
        
        output_dir = Path("data/size_charts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "size_charts.json"
        with open(output_file, 'w') as f:
            json.dump(self.size_charts, f, indent=4)
        
        self.logger.info(f"Size charts saved to {output_file}")
        print(f"\nSize charts saved to {output_file}")

def main():
    """Main function to collect size charts"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        collector = SizeChartCollector()
        
        # Collect size charts from all retailers
        collector.collect_all_retailers()
        
        # Save the results
        collector.save_size_charts()
        
        logger.info("Size chart collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 