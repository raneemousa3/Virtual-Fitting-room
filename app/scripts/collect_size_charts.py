import logging
from pathlib import Path
import sys

# Add the parent directory to the path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from services.size_chart_collector import SizeChartCollector

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to collect size charts"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize collector
        collector = SizeChartCollector()
        
        # Collect size charts
        logger.info("Starting size chart collection...")
        size_charts = collector.collect_all_size_charts()
        
        # Print summary
        logger.info("\nSize Chart Collection Summary:")
        for brand, data in size_charts.items():
            logger.info(f"\n{brand.upper()}:")
            if data:
                logger.info(f"  Sizes collected: {len(data)}")
                # Print first size as example
                first_size = next(iter(data.values()))
                logger.info(f"  Example measurements: {first_size}")
            else:
                logger.info("  No data collected")
        
    except Exception as e:
        logger.error(f"Error collecting size charts: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 