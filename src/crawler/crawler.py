import asyncio
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Set
import os
from pathlib import Path

from .generator import TopicGenerator
from .validator import TopicValidator
from .graph import RefusalGraph
from ..utils.model_api import ModelAPI

class RefusalCrawler:
    def __init__(self, config_path: str):
        # Store config path and setup logging first
        self.config_path = config_path
        self._setup_logging()
        
        # Load configuration
        self.logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Print initial setup information
        self.logger.info("=" * 50)
        self.logger.info("Starting Refusal Crawler")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {self.config['model']['name']}")
        self.logger.info(f"Initial seed topics: {self.config['crawler']['seed_topics']}")
        self.logger.info(f"Max iterations: {self.config['crawler']['max_iterations']}")
        self.logger.info(f"Max hours: {self.config['crawler']['max_hours']}")
        self.logger.info("=" * 50)
            
        # Initialize components
        self.model_api = ModelAPI(config=self.config)
        self.generator = TopicGenerator(self.model_api, self.config)
        self.validator = TopicValidator(self.model_api, self.config)
        self.graph = RefusalGraph()
        
        # Initialize with seed topics
        self._initialize_seed_topics()
        
    def _initialize_seed_topics(self):
        """Initialize the crawler with seed topics"""
        self.logger.info("Initializing seed topics...")
        for topic in self.config['crawler']['seed_topics']:
            self.graph.add_topic(topic)
            self.validator.add_validated_topic(topic)
        self.logger.info(f"Initialized with {len(self.config['crawler']['seed_topics'])} seed topics")
        
    async def run(self):
        """Run the crawler"""
        start_time = datetime.now()
        iteration = 0
        max_iterations = self.config['crawler']['max_iterations']
        max_hours = self.config['crawler']['max_hours']
        
        self.logger.info("Beginning crawl process...")
        self.logger.info(f"Current time: {start_time}")
        
        while iteration < max_iterations:
            # Check time limit
            elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
            if elapsed_hours > max_hours:
                self.logger.info(f"Reached time limit of {max_hours} hours")
                break
                
            self.logger.info("\n" + "=" * 30)
            self.logger.info(f"Starting iteration {iteration}")
            self.logger.info(f"Elapsed time: {elapsed_hours:.2f} hours")
            self.logger.info("\n" + "=" * 30)
            
            # Generate new topics using thought forcing
            new_topics = await self.generator.generate_new_topics(iteration)
            
            if not new_topics and len(self.generator.backlog) == 0:
                self.logger.info("No new topics found and backlog is empty. Stopping crawler.")
                break
                
            if not new_topics:
                self.logger.warning("No new topics generated in this iteration")
                self._log_statistics(iteration, [])
                iteration += 1
                continue
                
            self.logger.info(f"Generated {len(new_topics)} potential new topics")
            
            # Process and validate new topics
            valid_topics = await self._process_new_topics(new_topics)
            
            # Update graph with valid topics
            if valid_topics:
                self._update_graph(valid_topics, iteration)
                self._save_progress(iteration)
                
            self._log_statistics(iteration, valid_topics)
            iteration += 1
            
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Crawl completed")
        self.logger.info(f"Total runtime: {(datetime.now() - start_time).total_seconds() / 3600:.2f} hours")
        self.logger.info(f"Total topics discovered: {len(self.validator.get_known_topics())}")
        self.logger.info("=" * 50)
        
        self._save_progress("final")
        
    async def _process_new_topics(self, topics: List[str]) -> List[str]:
        """Process and validate new topics"""
        valid_topics = []
        
        for topic in topics:
            self.logger.info(f"\nValidating topic: {topic}")
            
            # Check for duplicates
            is_duplicate, existing_topic = await self.validator.is_duplicate(topic)
            if is_duplicate:
                self.logger.info(f"  ❌ Duplicate of existing topic '{existing_topic}'")
                continue
                
            # Add to valid topics
            self.logger.info(f"  ✅ New valid topic: {topic}")
            valid_topics.append(topic)
            self.validator.add_validated_topic(topic)
            
        return valid_topics
        
    def _update_graph(self, topics: List[str], iteration: int):
        """Update graph with new topics"""
        self.logger.info(f"\nUpdating graph with {len(topics)} new topics")
        for topic in topics:
            # Find parent topic (using first topic in current generation)
            parent = next(iter(self.generator.current_topics))
            self.logger.info(f"  Adding: {topic} (parent: {parent})")
            
            self.graph.add_topic(
                topic,
                parent_topic=parent,
                metadata={
                    'iteration': iteration,
                    'parent': parent,
                    'discovery_time': datetime.now().isoformat()
                }
            )
            
    def _log_statistics(self, iteration: int, valid_topics: List[str]):
        """Log statistics for the current iteration"""
        total_topics = len(self.validator.get_known_topics())
        self.logger.info("\nCurrent Statistics:")
        self.logger.info(f"Total topics in graph: {total_topics}")
        self.logger.info(f"Topics found this iteration: {len(valid_topics)}")
        self.logger.info(f"Backlog size: {len(self.generator.backlog)}")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("data/crawl_history")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing handlers to avoid duplicate logging
        logging.getLogger().handlers = []
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "crawler.log"),
                logging.StreamHandler()  # This will show logs in console too
            ]
        )
        
        # Create logger for this class
        self.logger = logging.getLogger("RefusalCrawler")
        self.logger.setLevel(logging.INFO)
        
    def _save_progress(self, iteration):
        """Save current progress"""
        # Create directories if they don't exist
        for dir_path in ["data/graphs", "data/crawl_history"]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Save graph
        graph_path = f"data/graphs/refusal_graph_{iteration}.json"
        self.graph.save_graph(graph_path)
        self.logger.info(f"Saved graph to {graph_path}")
        
        # Save known topics list
        topics_path = f"data/crawl_history/known_topics_{iteration}.txt"
        with open(topics_path, 'w') as f:
            for topic in self.validator.get_known_topics():
                f.write(f"{topic}\n")
        self.logger.info(f"Saved known topics to {topics_path}") 