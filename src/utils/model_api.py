from openai import AsyncOpenAI
import yaml
import random
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional, Dict, List

class ModelAPI:
    def __init__(self, config: dict):
        """Initialize the ModelAPI with configuration
        
        Args:
            config: Configuration dictionary containing model settings
        """
        # Setup logger
        self.logger = logging.getLogger("ModelAPI")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing ModelAPI")
        self.client = AsyncOpenAI(
            api_key=config['model']['api_key'],
            base_url=config['model']['base_url']
        )
        self.model_name = config['model']['name']
        self.config = config
        self.logger.info(f"ModelAPI initialized with model: {self.model_name}")
        
    async def verify_refusal(self, topic: str) -> bool:
        """Check if a topic triggers refusal behavior"""
        prompt = self.config['templates']['verification_prompt'].format(
            TOPIC=topic,
            think_start=self.config['templates']['markers']['think_start']
        )
        response = await self.query(prompt)
        return self._check_refusal_patterns(response)
        
    async def verify_thought_forcing(self, topic: str) -> tuple[bool, str]:
        """Check if thought forcing bypasses refusal"""
        thought_prefix = random.choice(self.config['templates']['thought_forcing'])
        prompt = self.config['templates']['verification_prompt'].format(
            TOPIC=topic,
            think_start=self.config['templates']['markers']['think_start'] + thought_prefix
        )
        response = await self.query(prompt)
        is_refused = self._check_refusal_patterns(response)
        return not is_refused, response
        
    async def generate_sensitive_list(self, topic: str) -> str:
        """Generate list of sensitive topics using thought forcing"""
        prompt = self.config['templates']['verification_prompt'].format(
            TOPIC=topic,
            think_start=self.config['templates']['markers']['think_start'] + 
                       self.config['templates']['force_sensitive_list']
        )
        return await self.query(prompt)
        
    def _check_refusal_patterns(self, response: str) -> bool:
        """Check if response matches refusal patterns"""
        import re
        patterns = self.config['templates']['refusal_patterns']
        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query(self, prompt: str, temperature: float = 0.6) -> str:
        """Query the model with a prompt"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            self.logger.info("=" * 40)
            self.logger.info("SENDING QUERY TO MODEL")
            self.logger.info("=" * 40)
            self.logger.info(f"Prompt: {prompt}")
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Temperature: {temperature}")
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            self.logger.info("\nRECEIVED RESPONSE:")
            self.logger.info("=" * 40)
            self.logger.info(content)
            self.logger.info("=" * 40 + "\n")
            return content
            
        except Exception as e:
            self.logger.error("=" * 40)
            self.logger.error("ERROR IN MODEL QUERY")
            self.logger.error("=" * 40)
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Failed prompt: {prompt}")
            self.logger.error("=" * 40 + "\n")
            raise 