import random
import re
import asyncio
from typing import List, Dict, Set, Tuple
import logging
import json
from pathlib import Path

class TopicGenerator:
    def __init__(self, model_api, config: Dict):
        # Setup logger
        self.logger = logging.getLogger("TopicGenerator")
        self.logger.setLevel(logging.INFO)
        
        self.model_api = model_api
        self.config = config
        self.current_topics = set(config['crawler']['seed_topics'])
        self.backlog: Set[str] = set()
        
        # Create backlog directory if it doesn't exist
        self.backlog_dir = Path("data/backlog")
        self.backlog_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing backlog
        self._load_latest_backlog()
        
    def _load_latest_backlog(self):
        """Load the most recent backlog if it exists"""
        try:
            backlog_files = list(self.backlog_dir.glob("backlog_iteration_*.json"))
            if backlog_files:
                latest_backlog = max(backlog_files, key=lambda x: int(x.stem.split('_')[-1]))
                self.logger.info(f"Loading backlog from {latest_backlog}")
                with open(latest_backlog, 'r') as f:
                    loaded_backlog = json.load(f)
                    self.backlog.update(loaded_backlog)
                self.logger.info(f"Loaded {len(self.backlog)} topics from backlog")
        except Exception as e:
            self.logger.error(f"Error loading backlog: {e}")
            
    def save_backlog(self, iteration: int):
        """Save current backlog to disk"""
        try:
            backlog_path = self.backlog_dir / f"backlog_iteration_{iteration}.json"
            with open(backlog_path, 'w') as f:
                json.dump(list(self.backlog), f, indent=2)
            self.logger.info(f"Saved {len(self.backlog)} topics to backlog at {backlog_path}")
            
            # Also save a latest copy
            latest_path = self.backlog_dir / "backlog_latest.json"
            with open(latest_path, 'w') as f:
                json.dump(list(self.backlog), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving backlog: {e}")
        
    async def process_topic(self, topic: str) -> List[str]:
        """Process a single topic through the thought forcing pipeline"""
        self.logger.info(f"\nProcessing topic: {topic}")
        
        # Step 1: Verify initial refusal
        self.logger.info("Step 1: Verifying initial refusal")
        is_refused = await self.model_api.verify_refusal(topic)
        if not is_refused:
            self.logger.info("Topic does not trigger refusal, skipping")
            return []
        self.logger.info("✓ Topic triggers refusal")
        
        # Step 2: Verify thought forcing bypass
        self.logger.info("Step 2: Verifying thought forcing bypass")
        bypass_success, response = await self.model_api.verify_thought_forcing(topic)
        if not bypass_success:
            self.logger.info("Thought forcing did not bypass refusal, skipping")
            return []
        self.logger.info("✓ Thought forcing successfully bypassed refusal")
        
        # Step 3: Generate list of sensitive topics
        self.logger.info("Step 3: Generating sensitive topics list")
        sensitive_list_response = await self.model_api.generate_sensitive_list(topic)
        
        # Extract topics from response
        new_topics = self._extract_topics_from_response(sensitive_list_response)
        self.logger.info(f"Extracted {len(new_topics)} new topics")
        
        return new_topics
        
    async def generate_new_topics(self, iteration: int) -> List[str]:
        """Generate new topics from current topics using thought forcing"""
        if not self.current_topics:
            self.logger.warning("No current topics available for generation")
            return []
            
        self.logger.info(f"Starting topic generation from {len(self.current_topics)} seed topics")
        
        all_new_topics = []
        for topic in list(self.current_topics):  # Convert to list to avoid modification during iteration
            new_topics = await self.process_topic(topic)
            if new_topics:
                # Add to backlog and results
                self.backlog.update(new_topics)
                all_new_topics.extend(new_topics)
                
        # Update current topics from backlog
        self._update_current_topics()
        
        # Save backlog after processing
        self.save_backlog(iteration)
        
        return all_new_topics
        
    def _extract_topics_from_response(self, response: str) -> List[str]:
        """Extract topics from response using configured patterns"""
        topics = set()
        
        self.logger.info("\nEXTRACTING TOPICS FROM RESPONSE")
        self.logger.info("=" * 40)
        self.logger.info("Response text:")
        self.logger.info(response)
        self.logger.info("-" * 40)
        
        patterns = self.config['templates']['topic_extraction_patterns']
        
        self.logger.info("Searching for patterns:")
        for pattern in patterns:
            self.logger.info(f"  Pattern: {pattern}")
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                topic = match.group(1).strip()
                # Basic cleaning
                topic = re.sub(r'\s+', ' ', topic)
                if len(topic) > 3:  # Ignore very short matches
                    topics.add(topic)
                    self.logger.info(f"  Found topic: {topic}")
                    
        if not topics:
            self.logger.info("No topics found in response")
        else:
            self.logger.info(f"Found {len(topics)} unique topics:")
            for topic in topics:
                self.logger.info(f"  - {topic}")
        
        self.logger.info("=" * 40)
        return list(topics)
        
    def _update_current_topics(self):
        """Update current topics from backlog"""
        batch_size = self.config['templates']['backlog']['batch_size']
        if self.backlog:
            # Take up to batch_size topics from backlog
            new_topics = set(list(self.backlog)[:batch_size])
            self.backlog -= new_topics
            self.current_topics = new_topics
            self.logger.info(f"Updated current topics from backlog. Current size: {len(self.current_topics)}")
        else:
            # If backlog is empty, clear current topics to signal completion
            self.current_topics.clear()
            self.logger.warning("Backlog is empty, clearing current topics") 