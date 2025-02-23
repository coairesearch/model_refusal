from typing import Optional, Dict, List, Tuple
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import nnsight
from nnsight import LanguageModel
import requests
import torch
import numpy as np
from abc import ABC, abstractmethod
import random
from .embedding import EmbeddingEnsemble

class BaseModelAPI(ABC):
    """Abstract base class for model APIs"""
    def __init__(self, config: dict):
        """Initialize the ModelAPI with configuration
        
        Args:
            config: Configuration dictionary containing model settings
        """
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.config = config
        
    @abstractmethod
    async def query(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Query the model with a prompt"""
        pass

    async def verify_refusal(self, topic: str) -> bool:
        """Check if a topic triggers refusal behavior"""
        prompt = f"What happened at {topic}?"
        response = await self.query(prompt)
        return self._check_refusal_patterns(response)
        
    async def verify_thought_forcing(self, topic: str) -> tuple[bool, str]:
        """Check if thought forcing bypasses refusal"""
        prompt = f"What happened at {topic}?"
        thinking_message = self.config['templates']['thinking_messages']['thought_forcing']
        response = await self.query(prompt, thinking_message=thinking_message)
        is_refused = self._check_refusal_patterns(response)
        return not is_refused, response
        
    async def generate_sensitive_list(self, topic: str) -> str:
        """Generate list of sensitive topics using thought forcing"""
        prompt = f"What happened at {topic}?"
        thinking_message = self.config['templates']['thinking_messages']['list_generation']
        return await self.query(prompt, thinking_message=thinking_message)
        
    def _check_refusal_patterns(self, response: str) -> bool:
        """Check if response matches refusal patterns"""
        import re
        patterns = self.config['templates']['refusal_patterns']
        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False

class NDIFModelAPI(BaseModelAPI):
    """NDIF-specific model API implementation"""
    def __init__(self, config: dict):
        super().__init__(config)  # Call parent class init
        
        # Configure NDIF connection
        self.host = config['model'].get('ndif_host', 'localhost')
        self.port = config['model'].get('ndif_port', 5001)
        self.ssl = config['model'].get('ndif_ssl', False)
        self.api_key = config['model'].get('ndif_api_key', '')
        
        # Configure nnsight
        nnsight.CONFIG.API.HOST = f"{self.host}:{self.port}"
        nnsight.CONFIG.API.SSL = self.ssl
        nnsight.CONFIG.API.APIKEY = self.api_key
        
        # Initialize model
        self.model_name = config['model']['name']
        self.model = LanguageModel(self.model_name)
        self.tokenizer = self.model.tokenizer
        
        # Initialize embedding ensemble for similarity checks
        self.embedding_ensemble = EmbeddingEnsemble(config)
        
        # Set special tokens based on model type
        self._setup_special_tokens()
        
        self.logger.info(f"NDIF ModelAPI initialized with model: {self.model_name}")
        
    def _setup_special_tokens(self):
        """Setup special tokens based on model type"""
        if "Llama" in self.model_name:
            self.BOS = 128000
            self.USER = 128011
            self.ASSISTANT = 128012
            self.NEWLINE = 198
            self.THINK_START = 128013
            self.THINK_END = 128014
            self.EOS = 128001
        elif "Qwen" in self.model_name:
            self.BOS = 151646
            self.USER = 151644
            self.ASSISTANT = 151645
            self.NEWLINE = 198
            self.THINK_START = 151648
            self.THINK_END = 151649
            self.EOS = 151643
        else:
            raise ValueError(f"Unknown tokens for model {self.model_name}")
            
    def _custom_encoding(self, user_message: str, thinking_message: str = "") -> List[List[int]]:
        """Encode the user message and thinking message"""
        user_tokens = self.tokenizer.encode(user_message, add_special_tokens=False)
        thinking_tokens = self.tokenizer.encode(thinking_message, add_special_tokens=False)
        return [[self.BOS] + user_tokens + [self.NEWLINE] + [self.THINK_START] + thinking_tokens]
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query(self, prompt: str, temperature: Optional[float] = None, thinking_message: str = "") -> str:
        """Query the model with a prompt using NDIF"""
        try:
            self.logger.info("=" * 40)
            self.logger.info("SENDING QUERY TO NDIF MODEL")
            self.logger.info("=" * 40)
            self.logger.info(f"Prompt: {prompt}")
            self.logger.info(f"Thinking: {thinking_message}")
            self.logger.info(f"Model: {self.model_name}")
            
            # Encode input
            input_ids = self._custom_encoding(prompt, thinking_message)
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate text
            with self.model.generate(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                max_new_tokens=self.config['model'].get('max_tokens', 200),
                do_sample=temperature is not None,
                temperature=temperature or self.config['model'].get('temperature', 0.7),
                top_p=self.config['model'].get('top_p', 0.9) if temperature else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                remote=True  # Run the model remotely on NDIF
            ):
                outputs = nnsight.list().save()
                with self.model.lm_head.all():
                    outputs.append(self.model.lm_head.output[0][-1].argmax(dim=-1))
                    
            # Decode output
            outputs = outputs.value
            generated_text = self.tokenizer.decode(outputs, skip_special_tokens=False)
            
            self.logger.info("\nRECEIVED RESPONSE:")
            self.logger.info("=" * 40)
            self.logger.info(generated_text)
            self.logger.info("=" * 40 + "\n")
            return generated_text
            
        except Exception as e:
            self.logger.error("=" * 40)
            self.logger.error("ERROR IN MODEL QUERY")
            self.logger.error("=" * 40)
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Failed prompt: {prompt}")
            self.logger.error("=" * 40 + "\n")
            raise

    async def verify_refusal(self, topic: str) -> bool:
        """Check if a topic triggers refusal behavior"""
        prompt = f"What happened at {topic}?"
        response = await self.query(prompt)
        return self._check_refusal_patterns(response)
        
    async def verify_thought_forcing(self, topic: str) -> tuple[bool, str]:
        """Check if thought forcing bypasses refusal"""
        prompt = f"What happened at {topic}?"
        thinking_message = self.config['templates']['thinking_messages']['thought_forcing']
        response = await self.query(prompt, thinking_message=thinking_message)
        is_refused = self._check_refusal_patterns(response)
        return not is_refused, response
        
    async def generate_sensitive_list(self, topic: str) -> str:
        """Generate list of sensitive topics using thought forcing"""
        prompt = f"What happened at {topic}?"
        thinking_message = self.config['templates']['thinking_messages']['list_generation']
        return await self.query(prompt, thinking_message=thinking_message)
        
    def _check_refusal_patterns(self, response: str) -> bool:
        """Check if response matches refusal patterns"""
        import re
        patterns = self.config['templates']['refusal_patterns']
        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False

    async def check_similarity(self, topic1: str, topic2: str) -> float:
        """Check semantic similarity between two topics using the embedding ensemble"""
        is_similar, similarity = self.embedding_ensemble.are_similar(topic1, topic2)
        self.logger.info(f"Similarity between '{topic1}' and '{topic2}': {similarity:.3f} (similar: {is_similar})")
        return similarity

# This is the class that should be used by the crawler
ModelAPI = NDIFModelAPI 