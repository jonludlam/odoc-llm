#!/usr/bin/env python3
"""
Semantic Search for OCaml Modules

This script embeds user queries using Qwen3-Embedding-0.6B and finds the most
semantically similar OCaml modules from the package embeddings dataset.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from popularity_scorer import PopularityScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Workaround for transformers 4.52.4 bug with ALL_PARALLEL_STYLES
import transformers.modeling_utils
if transformers.modeling_utils.ALL_PARALLEL_STYLES is None:
    # Define expected parallel styles from transformers library
    transformers.modeling_utils.ALL_PARALLEL_STYLES = ["colwise", "rowwise"]
    logger.warning("Patched ALL_PARALLEL_STYLES bug in transformers library")


class QueryEmbedder:
    """Handles embedding of user queries using local transformer model or embedding server."""
    
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', use_server: bool = False, server_url: str = "http://localhost:8000"):
        """Initialize the query embedder with the specified model."""
        logger.info(f"Initializing embedder with model: {model_name}, use_server: {use_server}")
        self.model_name = model_name
        self.use_server = use_server
        self.server_url = server_url
        
        if not use_server:
            # Load model locally
            logger.info(f"Loading embedding model locally: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
        else:
            logger.info(f"Using embedding server at {server_url}")
            self.tokenizer = None
            self.model = None
            
        self.max_length = 8192
        
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last token from the hidden states."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format the query with task instruction."""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query and return the normalized embedding vector.
        
        Args:
            query: The user's search query
            
        Returns:
            Normalized embedding vector as numpy array
        """
        if self.use_server:
            # Use embedding server
            import requests
            
            # Task description for OCaml module search
            task = 'Given a programming task or functionality description, retrieve relevant OCaml modules that provide that functionality'
            formatted_query = self.get_detailed_instruct(task, query)
            
            try:
                response = requests.post(
                    f"{self.server_url}/embedding",
                    json={"content": formatted_query},
                    headers={"Content-Type": "application/json"},
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Extract embedding from response
                    if data and len(data) > 0 and "embedding" in data[0]:
                        embedding = np.array(data[0]["embedding"][0], dtype=np.float32)
                        # Normalize
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        return embedding
                    else:
                        raise ValueError(f"Unexpected response format: {data}")
                else:
                    raise ValueError(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Failed to get embedding from server: {e}")
                raise
        else:
            # Use local model
            # Task description for OCaml module search
            task = 'Given a programming task or functionality description, retrieve relevant OCaml modules that provide that functionality'
            
            # Format query with instruction
            formatted_query = self.get_detailed_instruct(task, query)
            
            # Tokenize
            batch_dict = self.tokenizer(
                [formatted_query],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            # Move to same device as model
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
            # Convert to numpy and return
            return embeddings.cpu().numpy()[0]


class PackageEmbeddingLoader:
    """Handles loading and managing package embeddings from the dataset."""
    
    def __init__(self, embeddings_dir: Path):
        """Initialize the loader with the embeddings directory."""
        self.embeddings_dir = embeddings_dir
        self.packages_dir = embeddings_dir / "packages"
        
        if not self.packages_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {self.packages_dir}")
            
        # Build package index
        self.package_index = self._build_package_index()
        logger.info(f"Found {len(self.package_index)} packages with embeddings")
        
    def _build_package_index(self) -> Dict[str, Dict]:
        """Build an index of available packages and their metadata."""
        index = {}
        
        for package_dir in self.packages_dir.iterdir():
            if not package_dir.is_dir():
                continue
                
            embeddings_file = package_dir / "embeddings.npz"
            metadata_file = package_dir / "metadata.json"
            
            if embeddings_file.exists() and metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    index[package_dir.name] = {
                        'embeddings_path': embeddings_file,
                        'metadata_path': metadata_file,
                        'metadata': metadata,
                        'num_modules': len(metadata.get('modules', []))
                    }
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {package_dir.name}: {e}")
                    
        return index
    
    def load_all_embeddings(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load all embeddings into memory and return consolidated arrays.
        
        Returns:
            Tuple of (embeddings_array, module_metadata_list)
        """
        logger.info("Loading all embeddings into memory...")
        start_time = time.time()
        
        all_embeddings = []
        all_metadata = []
        
        for package_name, package_info in self.package_index.items():
            try:
                # Load embeddings
                embeddings_data = np.load(package_info['embeddings_path'])
                embeddings = embeddings_data['embeddings']
                
                # Get modules from metadata
                modules = package_info['metadata'].get('modules', [])
                
                # Ensure embeddings and metadata are aligned
                if len(embeddings) != len(modules):
                    logger.warning(f"Mismatch in {package_name}: {len(embeddings)} embeddings vs {len(modules)} modules in metadata")
                    # Take the minimum to avoid index errors
                    num_to_use = min(len(embeddings), len(modules))
                    embeddings = embeddings[:num_to_use]
                    modules = modules[:num_to_use]
                
                # Add embeddings
                all_embeddings.append(embeddings)
                
                # Add metadata for each module
                for module_info in modules:
                    module_metadata = {
                        'package': package_name,
                        'module_path': module_info['module_path'],
                        'description': module_info['description'],
                        'description_length': module_info['description_length'],
                        'index_in_package': module_info['index']
                    }
                    # Add library information if available
                    if 'library' in module_info:
                        module_metadata['library'] = module_info['library']
                    all_metadata.append(module_metadata)
                    
            except Exception as e:
                logger.warning(f"Failed to load embeddings for {package_name}: {e}")
                
        # Concatenate all embeddings
        if all_embeddings:
            consolidated_embeddings = np.vstack(all_embeddings)
            logger.info(f"Loaded {len(consolidated_embeddings)} embeddings in {time.time() - start_time:.2f}s")
            return consolidated_embeddings, all_metadata
        else:
            raise RuntimeError("No embeddings could be loaded")


class SemanticSearch:
    """Main semantic search engine that combines query embedding and similarity search."""
    
    def __init__(self, embeddings_dir: Path, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', 
                 use_popularity: bool = True, package_descriptions_dir: Path = None,
                 package_description_embeddings_dir: Path = None, server_url: str = "http://localhost:8000"):
        """Initialize the semantic search engine."""
        # Detect if we should use server based on model name
        use_server = "8B" in model_name or "7B" in model_name
        self.query_embedder = QueryEmbedder(model_name, use_server=use_server, server_url=server_url)
        self.embedding_loader = PackageEmbeddingLoader(embeddings_dir)
        self.package_description_embeddings_dir = package_description_embeddings_dir
        
        # Load all embeddings into memory
        self.embeddings, self.metadata = self.embedding_loader.load_all_embeddings()
        
        # Initialize popularity scorer if enabled
        self.use_popularity = use_popularity
        self.popularity_scorer = PopularityScorer() if use_popularity else None
        
        # Load package descriptions for boosting
        self.package_descriptions = {}
        if package_descriptions_dir is None:
            package_descriptions_dir = Path("package-descriptions")
        self.load_package_descriptions(package_descriptions_dir)
        
        # Load pre-computed package description embeddings
        self.package_description_embeddings = {}
        if package_description_embeddings_dir is None:
            package_description_embeddings_dir = Path("package-description-embeddings")
        self.load_package_description_embeddings(package_description_embeddings_dir)
        
        # Load package reverse dependencies data
        self.package_revdeps = {}
        self.load_package_revdeps()
        
        logger.info(f"Search engine ready with {len(self.embeddings)} modules")
        
    def load_package_descriptions(self, descriptions_dir: Path) -> None:
        """Load package descriptions for boosting module rankings."""
        if not descriptions_dir.exists():
            logger.warning(f"Package descriptions directory not found: {descriptions_dir}")
            return
            
        loaded_count = 0
        for json_file in descriptions_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'package' in data and 'description' in data:
                    self.package_descriptions[data['package']] = {
                        'description': data['description'],
                        'version': data.get('version', 'Unknown')
                    }
                    loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load package description {json_file}: {e}")
                
        logger.info(f"Loaded {loaded_count} package descriptions")
    
    def load_package_description_embeddings(self, embeddings_dir: Path) -> None:
        """Load pre-computed package description embeddings."""
        if not embeddings_dir.exists():
            logger.warning(f"Package description embeddings directory not found: {embeddings_dir}")
            return
            
        packages_dir = embeddings_dir / "packages"
        if not packages_dir.exists():
            logger.warning(f"Package description embeddings packages directory not found: {packages_dir}")
            return
            
        loaded_count = 0
        for package_dir in packages_dir.iterdir():
            if not package_dir.is_dir():
                continue
                
            try:
                # Load embedding
                embeddings_file = package_dir / "embeddings.npz"
                if embeddings_file.exists():
                    data = np.load(embeddings_file)
                    # Package embeddings have shape (1, embedding_dim)
                    package_embedding = data['embeddings'][0]
                    package_name = package_dir.name
                    self.package_description_embeddings[package_name] = package_embedding
                    loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load package description embedding for {package_dir.name}: {e}")
                
        logger.info(f"Loaded {loaded_count} pre-computed package description embeddings")
    
    def load_package_revdeps(self) -> None:
        """Load package reverse dependencies data for popularity scoring."""
        revdeps_file = Path("package-revdeps-counts.json")
        
        if not revdeps_file.exists():
            logger.warning(f"Package reverse dependencies file not found: {revdeps_file}")
            return
            
        try:
            with open(revdeps_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract dependencies array and convert to lookup dict
            if 'dependencies' in data and isinstance(data['dependencies'], list):
                for entry in data['dependencies']:
                    if 'package' in entry and 'reverse_dependencies' in entry:
                        package_name = entry['package']
                        revdep_count = entry['reverse_dependencies']
                        self.package_revdeps[package_name] = revdep_count
                        
                logger.info(f"Loaded {len(self.package_revdeps)} package reverse dependency counts")
            else:
                logger.warning("Unexpected format in package reverse dependencies file")
                
        except Exception as e:
            logger.warning(f"Failed to load package reverse dependencies: {e}")
    
    def calculate_package_boost(self, query: str, package_names: List[str]) -> Dict[str, float]:
        """Calculate package description similarity boost for modules using pre-computed embeddings."""
        if not self.package_description_embeddings:
            logger.info("No pre-computed package description embeddings available, skipping package boost")
            return {}
            
        # Get query embedding once
        query_embedding = self.query_embedder.embed_query(query)
        
        # Only calculate for unique packages that have pre-computed embeddings
        unique_packages = set(package_names)
        packages_with_embeddings = [p for p in unique_packages if p in self.package_description_embeddings]
        
        logger.info(f"Calculating package boosts for {len(packages_with_embeddings)} packages using pre-computed embeddings")
        
        package_boosts = {}
        for package in unique_packages:
            if package in self.package_description_embeddings:
                # Use pre-computed package description embedding
                package_embedding = self.package_description_embeddings[package]
                
                # Calculate similarity
                similarity = float(np.dot(query_embedding, package_embedding))
                package_boosts[package] = similarity
            else:
                package_boosts[package] = 0.0
                
        logger.info(f"Calculated package boosts: {len(package_boosts)} packages")
        
        return package_boosts
        
    def search_packages_only(self, query: str, top_k: int = 5, use_popularity_ranking: bool = False, 
                           popularity_weight: float = 0.3) -> List[Dict]:
        """
        Search for packages based only on their descriptions.
        
        Args:
            query: User's search query
            top_k: Number of top results to return
            use_popularity_ranking: Whether to incorporate reverse dependency counts in ranking
            popularity_weight: Weight for popularity in combined score (0.0-1.0)
            
        Returns:
            List of dictionaries containing package information and similarity scores
        """
        logger.info(f"Searching packages only for: '{query}' (popularity ranking: {use_popularity_ranking})")
        start_time = time.time()
        
        if not self.package_description_embeddings:
            raise RuntimeError("No package description embeddings loaded. Cannot perform package-only search.")
        
        # Embed the query
        query_embedding = self.query_embedder.embed_query(query)
        
        # Calculate similarities with package descriptions
        package_results = []
        for package_name, package_embedding in self.package_description_embeddings.items():
            similarity = float(np.dot(query_embedding, package_embedding))
            
            # Get package description text if available
            package_data = self.package_descriptions.get(package_name, {})
            description = package_data.get('description', 'No description available') if isinstance(package_data, dict) else package_data
            version = package_data.get('version', 'Unknown') if isinstance(package_data, dict) else 'Unknown'
            
            # Get reverse dependency count (popularity)
            revdep_count = self.package_revdeps.get(package_name, 0)
            
            # Calculate combined score if using popularity ranking
            combined_score = similarity
            if use_popularity_ranking and self.package_revdeps:
                # Normalize popularity score using logit transformation
                max_revdeps = max(self.package_revdeps.values()) if self.package_revdeps else 1
                
                # First normalize to [0, 1] range
                linear_normalized = revdep_count / max_revdeps if max_revdeps > 0 else 0.0
                
                # Apply logit transformation: logit(p) = log(p / (1 - p))
                # Add small epsilon to avoid division by zero and ensure p is in (0, 1)
                epsilon = 1e-6
                p = linear_normalized * (1 - 2 * epsilon) + epsilon
                logit_popularity = np.log(p / (1 - p))
                
                # Normalize logit values to [0, 1] range
                # logit(epsilon) to logit(1-epsilon) maps to approximately [-13.8, 13.8]
                min_logit = np.log(epsilon / (1 - epsilon))
                max_logit = np.log((1 - epsilon) / epsilon)
                normalized_popularity = (logit_popularity - min_logit) / (max_logit - min_logit)
                
                # Combine similarity and popularity
                combined_score = (1 - popularity_weight) * similarity + popularity_weight * normalized_popularity
            
            package_results.append({
                'package': package_name,
                'similarity_score': similarity,
                'combined_score': combined_score,
                'reverse_dependencies': revdep_count,
                'description': description,
                'version': version,
                'search_type': 'package_description'
            })
        
        # Sort by combined score (or similarity if not using popularity)
        sort_key = 'combined_score' if use_popularity_ranking else 'similarity_score'
        package_results.sort(key=lambda x: x[sort_key], reverse=True)
        results = package_results[:top_k]
        
        # Add rank information
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        search_time = time.time() - start_time
        logger.info(f"Package search completed in {search_time:.3f}s")
        
        return results

    def search(self, query: str, top_k: int = 5, popularity_weight: float = 0.3, 
              package_boost_weight: float = 0.2) -> List[Dict]:
        """
        Search for modules most similar to the given query.
        
        Args:
            query: User's search query
            top_k: Number of top results to return
            popularity_weight: Weight for popularity in combined score (0.0-1.0)
            package_boost_weight: Weight for package description boost (0.0-1.0)
            
        Returns:
            List of dictionaries containing module information and similarity scores
        """
        logger.info(f"Searching for: '{query}'")
        start_time = time.time()
        
        # Embed the query
        query_embedding = self.query_embedder.embed_query(query)
        
        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding)
        
        # Apply package boosting if enabled
        if self.package_descriptions and package_boost_weight > 0:
            # Get package names for all modules
            package_names = [self.metadata[i]['package'] for i in range(len(self.metadata))]
            
            # Calculate package boosts
            package_boosts = self.calculate_package_boost(query, package_names)
            
            # Apply package boost to similarities
            for i, package in enumerate(package_names):
                boost = package_boosts.get(package, 0.0)
                similarities[i] = similarities[i] * (1 - package_boost_weight) + boost * package_boost_weight
        
        # Get initial top results (retrieve more for reranking)
        initial_top_k = min(top_k * 3, len(similarities))  # Get 3x results for reranking
        top_indices = np.argsort(similarities)[-initial_top_k:][::-1]  # Descending order
        
        # Format initial results
        results = []
        for idx in top_indices:
            result = {
                'similarity_score': float(similarities[idx]),
                'package': self.metadata[idx]['package'],
                'module_path': self.metadata[idx]['module_path'],
                'description': self.metadata[idx]['description'],
                'description_length': self.metadata[idx]['description_length']
            }
            # Include library information if available
            if 'library' in self.metadata[idx]:
                result['library'] = self.metadata[idx]['library']
            results.append(result)
        
        # Apply popularity-based reranking if enabled
        if self.use_popularity and self.popularity_scorer:
            # Convert to tuple format expected by rerank_results
            result_tuples = [(r['package'], r['module_path'], r['description'], r['similarity_score']) 
                           for r in results]
            
            # Rerank with popularity
            reranked_tuples = self.popularity_scorer.rerank_results(result_tuples, popularity_weight)
            
            # Convert back to dict format and keep only top_k
            final_results = []
            for i, (package, module_path, description, score) in enumerate(reranked_tuples[:top_k]):
                # Get original similarity score for debugging
                original_score = None
                for orig_result in results:
                    if (orig_result['package'] == package and 
                        orig_result['module_path'] == module_path):
                        original_score = orig_result['similarity_score']
                        break
                
                # Get popularity score for this module
                popularity_score = self.popularity_scorer.get_popularity_score(module_path)
                
                # Debug: show what we're looking up
                if popularity_score > 0:
                    logger.info(f"Found popularity for {module_path}: {popularity_score:.4f}")
                else:
                    logger.debug(f"No popularity data for {module_path}")
                
                result = {
                    'rank': i + 1,
                    'similarity_score': score,
                    'original_similarity': original_score,
                    'popularity_score': popularity_score,
                    'package': package,
                    'module_path': module_path,
                    'description': description,
                }
                # Find original metadata to add extra fields
                for meta in self.metadata:
                    if meta['package'] == package and meta['module_path'] == module_path:
                        result['description_length'] = meta['description_length']
                        if 'library' in meta:
                            result['library'] = meta['library']
                        break
                final_results.append(result)
            results = final_results
        else:
            # No popularity scoring, just add ranks to final results
            results = results[:top_k]
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f}s")
        
        return results
    
    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between query and all module embeddings."""
        # Embeddings are already normalized, so dot product gives cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        return similarities


def format_results(results: List[Dict], format_type: str = 'text') -> str:
    """Format search results for display."""
    if format_type == 'json':
        return json.dumps(results, indent=2)
    
    elif format_type == 'text':
        output = []
        
        # Check if these are package-only results
        if results and results[0].get('search_type') == 'package_description':
            output.append(f"Top {len(results)} most relevant OCaml packages:\n")
            
            for result in results:
                output.append(f"#{result['rank']} - {result['package']} (v{result.get('version', 'Unknown')})")
                output.append(f"  Similarity: {result['similarity_score']:.4f}")
                
                # Show reverse dependencies count
                revdep_count = result.get('reverse_dependencies', 0)
                output.append(f"  Reverse Dependencies: {revdep_count}")
                
                # Show combined score if different from similarity
                if 'combined_score' in result and abs(result['combined_score'] - result['similarity_score']) > 0.001:
                    output.append(f"  Combined Score: {result['combined_score']:.4f}")
                
                output.append(f"  Description: {result['description']}")
                output.append("")
        else:
            # Original module search format
            output.append(f"Top {len(results)} most relevant OCaml modules:\n")
            
            for result in results:
                output.append(f"#{result['rank']} - {result['package']}: {result['module_path']}")
                output.append(f"  Similarity: {result['similarity_score']:.4f}")
                
                # Add popularity debugging info if available
                if 'popularity_score' in result:
                    output.append(f"  Original Similarity: {result.get('original_similarity', 'N/A'):.4f}")
                    output.append(f"  Popularity Score: {result['popularity_score']:.4f}")
                    
                output.append(f"  Description: {result['description']}")
                output.append("")
            
        return '\n'.join(output)
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Semantic search for OCaml modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python semantic_search.py "http server"
  python semantic_search.py "parse JSON data" --top-k 10
  python semantic_search.py "crypto hash function" --format json
  python semantic_search.py "JSON parsing" --packages-only --top-k 10
  python semantic_search.py "web framework" --packages-only --use-package-popularity --top-k 10
        """
    )
    
    parser.add_argument(
        'query',
        help='Search query describing the desired functionality'
    )
    
    parser.add_argument(
        '--embeddings-dir',
        type=Path,
        default=Path('package-embeddings'),
        help='Directory containing package embeddings (default: package-embeddings)'
    )
    
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-Embedding-0.6B',
        help='Embedding model to use (default: Qwen/Qwen3-Embedding-0.6B)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to return (default: 5)'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-popularity',
        action='store_true',
        help='Disable popularity-based ranking'
    )
    
    parser.add_argument(
        '--popularity-weight',
        type=float,
        default=0.3,
        help='Weight for popularity in ranking (0.0-1.0, default: 0.3)'
    )
    
    parser.add_argument(
        '--package-boost-weight',
        type=float,
        default=0.2,
        help='Weight for package description boost (0.0-1.0, default: 0.2)'
    )
    
    parser.add_argument(
        '--package-descriptions-dir',
        type=Path,
        default=Path('package-descriptions'),
        help='Directory containing package descriptions (default: package-descriptions)'
    )
    
    parser.add_argument(
        '--package-description-embeddings-dir',
        type=Path,
        default=Path('package-description-embeddings'),
        help='Directory containing pre-computed package description embeddings (default: package-description-embeddings)'
    )
    
    parser.add_argument(
        '--server-url',
        default='http://localhost:8000',
        help='URL for embedding server when using large models (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--packages-only',
        action='store_true',
        help='Search only package descriptions, not individual modules'
    )
    
    parser.add_argument(
        '--use-package-popularity',
        action='store_true',
        help='Use reverse dependency counts for package ranking (packages-only mode)'
    )
    
    parser.add_argument(
        '--package-popularity-weight',
        type=float,
        default=0.3,
        help='Weight for package popularity in ranking (0.0-1.0, default: 0.3)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize search engine
        search_engine = SemanticSearch(args.embeddings_dir, args.model, 
                                     use_popularity=not args.no_popularity,
                                     package_descriptions_dir=args.package_descriptions_dir,
                                     package_description_embeddings_dir=args.package_description_embeddings_dir,
                                     server_url=args.server_url)
        
        # Perform search
        if args.packages_only:
            results = search_engine.search_packages_only(
                args.query, 
                args.top_k,
                use_popularity_ranking=args.use_package_popularity,
                popularity_weight=args.package_popularity_weight
            )
        else:
            results = search_engine.search(args.query, args.top_k, 
                                         popularity_weight=args.popularity_weight,
                                         package_boost_weight=args.package_boost_weight)
        
        # Format and display results
        formatted_output = format_results(results, args.format)
        print(formatted_output)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())