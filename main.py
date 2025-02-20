import asyncio
import argparse
import logging
from pathlib import Path
from src.crawler.crawler import RefusalCrawler

async def setup_environment():
    """Setup the environment before running the crawler"""
    # Create necessary directories
    for dir_path in ["data/graphs", "data/crawl_history", "data/backlog"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/crawl_history/crawler.log'),
            logging.StreamHandler()
        ]
    )

async def main():
    """Main entry point for the refusal crawler"""
    parser = argparse.ArgumentParser(
        description='Run the refusal crawler with thought forcing approach'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()
    
    # Setup environment
    await setup_environment()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run crawler
    try:
        logging.info("Initializing RefusalCrawler...")
        crawler = RefusalCrawler(args.config)
        logging.info("Starting crawl process...")
        await crawler.run()
    except Exception as e:
        logging.error(f"Error during crawler execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 