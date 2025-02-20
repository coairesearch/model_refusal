import os
import logging
from pathlib import Path

# Setup CUDA environment variables first
def setup_cuda_env(config):
    cuda_device = config['embedding']['device']['cuda_device']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    
    if 'cuda_env_vars' in config['embedding']['device']:
        for var, value in config['embedding']['device']['cuda_env_vars'].items():
            if var != 'CUDA_VISIBLE_DEVICES':
                os.environ[var] = value
            logging.info(f"Set {var}={value}")

# Now import torch and other dependencies
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Optional, Tuple

class EmbeddingEnsemble:
    def __init__(self, config: Dict):
        self.config = config
        setup_cuda_env(config)  # Setup CUDA env before any torch operations
        self.device = self._setup_device()
        self.models = {}
        self.tokenizers = {}
        self.weights = config['embedding']['ensemble']['weights']
        self.similarity_threshold = config['embedding']['ensemble']['similarity_threshold']
        self.min_models_agreement = config['embedding']['ensemble']['min_models_agreement']
        
        # Initialize models
        self._initialize_models()
        
    def _get_available_cuda_devices(self) -> List[int]:
        """Get list of available CUDA devices"""
        available_devices = []
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logging.info(f"PyTorch sees {device_count} CUDA devices")
            for i in range(device_count):
                try:
                    with torch.cuda.device(i):
                        torch.tensor([1]).cuda()
                        mem_info = torch.cuda.get_device_properties(i).total_memory
                        logging.info(f"CUDA device {i} is available with {mem_info/1e9:.2f} GB memory")
                    available_devices.append(i)
                except Exception as e:
                    logging.warning(f"CUDA device {i} is not usable: {e}")
        return available_devices
        
    def _setup_device(self) -> torch.device:
        """Setup the device based on configuration and availability"""
        available_devices = self._get_available_cuda_devices()
        logging.info(f"Available CUDA devices after environment setup: {available_devices}")
        
        if torch.cuda.is_available():
            try:
                # Since we set CUDA_VISIBLE_DEVICES, the device should now be cuda:0
                device = torch.device('cuda:0')
                torch.cuda.set_device(device)
                
                # Print memory info
                mem_info = torch.cuda.get_device_properties(0)
                logging.info(f"Using CUDA device with {mem_info.total_memory/1e9:.2f} GB total memory")
                logging.info(f"Current memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                logging.info(f"Current memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                
                return device
            except Exception as e:
                logging.error(f"Error setting up CUDA device: {e}", exc_info=True)
                if self.config['embedding']['device']['fallback_to_cpu']:
                    logging.warning("Falling back to CPU")
                    return torch.device('cpu')
                raise
        
        logging.info("Using CPU device")
        return torch.device('cpu')
        
    def _initialize_models(self):
        """Initialize all embedding models"""
        models_config = self.config['embedding']['models']
        
        # Clear CUDA cache before loading models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"Cleared CUDA cache. Available memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        for model_key, model_config in models_config.items():
            try:
                if model_config['type'] == 'sentence_transformers':
                    model = SentenceTransformer(model_config['name'])
                    model.to(self.device)
                    self.models[model_key] = model
                    
                    # Log memory usage after loading each model
                    if torch.cuda.is_available():
                        logging.info(f"After loading {model_key}: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
                    
                elif model_config['type'] == 'transformers':
                    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
                    model = AutoModel.from_pretrained(model_config['name'])
                    model.to(self.device)
                    self.models[model_key] = model
                    self.tokenizers[model_key] = tokenizer
                    
                    # Log memory usage after loading each model
                    if torch.cuda.is_available():
                        logging.info(f"After loading {model_key}: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
                    
                logging.info(f"Successfully loaded {model_key} model: {model_config['name']}")
                
            except Exception as e:
                logging.error(f"Failed to load {model_key} model: {e}")
                if model_key in self.weights:
                    self._adjust_weights_after_failure(model_key)
                    
                # Try to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.info(f"Cleared CUDA cache after failure. Available memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
                    
    def _adjust_weights_after_failure(self, failed_model: str):
        """Redistribute weights if a model fails to load"""
        failed_weight = self.weights.pop(failed_model)
        total_remaining = sum(self.weights.values())
        
        if total_remaining > 0:
            # Redistribute the failed model's weight proportionally
            for model in self.weights:
                self.weights[model] *= (1 + failed_weight / total_remaining)
                
    def _get_embedding_sentence_transformers(self, 
                                          model: SentenceTransformer, 
                                          text: str, 
                                          batch_size: int) -> np.ndarray:
        """Get embeddings using sentence-transformers"""
        with torch.no_grad():
            embedding = model.encode(
                [text],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return embedding[0]
        
    def _get_embedding_transformers(self, 
                                 model: AutoModel, 
                                 tokenizer: AutoTokenizer, 
                                 text: str, 
                                 max_length: int) -> np.ndarray:
        """Get embeddings using transformers"""
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, 
                         truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding[0]
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get weighted average embedding from all models"""
        embeddings = {}
        total_weight = 0
        
        for model_key, model in self.models.items():
            try:
                model_config = self.config['embedding']['models'][model_key]
                
                if model_config['type'] == 'sentence_transformers':
                    embedding = self._get_embedding_sentence_transformers(
                        model,
                        text,
                        model_config['batch_size']
                    )
                else:  # transformers
                    embedding = self._get_embedding_transformers(
                        model,
                        self.tokenizers[model_key],
                        text,
                        model_config['max_length']
                    )
                    
                embeddings[model_key] = embedding
                total_weight += self.weights[model_key]
                
            except Exception as e:
                logging.error(f"Error getting embedding from {model_key}: {e}")
                
        if not embeddings:
            raise RuntimeError("No embeddings could be generated from any model")
            
        # Calculate weighted average
        weighted_embedding = np.zeros_like(next(iter(embeddings.values())))
        for model_key, embedding in embeddings.items():
            weighted_embedding += embedding * (self.weights[model_key] / total_weight)
            
        # Normalize
        return weighted_embedding / np.linalg.norm(weighted_embedding)
        
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2)
        
    def are_similar(self, text1: str, text2: str) -> Tuple[bool, float]:
        """Check if two texts are similar using ensemble voting"""
        similar_votes = 0
        similarities = []
        
        for model_key, model in self.models.items():
            try:
                model_config = self.config['embedding']['models'][model_key]
                
                if model_config['type'] == 'sentence_transformers':
                    emb1 = self._get_embedding_sentence_transformers(
                        model, text1, model_config['batch_size'])
                    emb2 = self._get_embedding_sentence_transformers(
                        model, text2, model_config['batch_size'])
                else:
                    emb1 = self._get_embedding_transformers(
                        model, self.tokenizers[model_key], 
                        text1, model_config['max_length'])
                    emb2 = self._get_embedding_transformers(
                        model, self.tokenizers[model_key], 
                        text2, model_config['max_length'])
                    
                similarity = self.compute_similarity(emb1, emb2)
                similarities.append(similarity)
                
                if similarity > self.similarity_threshold:
                    similar_votes += 1
                    
            except Exception as e:
                logging.error(f"Error computing similarity with {model_key}: {e}")
                
        # Calculate weighted average similarity
        if similarities:
            avg_similarity = np.average(similarities, 
                                      weights=[self.weights[m] for m in self.models.keys()])
        else:
            raise RuntimeError("No similarities could be computed from any model")
            
        return (similar_votes >= self.min_models_agreement, avg_similarity) 