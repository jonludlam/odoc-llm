#!/usr/bin/env python3
"""
Unified search across all OCaml packages (or specified subset).
Combines semantic search using embeddings with keyword search using BM25.
"""

import sys
import os

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import bm25s
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import requests


class UnifiedSearchEngine:
    """Combines semantic and keyword-based search for OCaml modules."""
    
    def __init__(self, embedding_dir: Path, index_dir: Path, 
                 api_url: str = 'http://localhost:8080', 
                 package_description_embeddings_dir: Path = None):
        self.embedding_dir = Path(embedding_dir)
        self.index_dir = Path(index_dir)
        self.package_description_embeddings_dir = Path(package_description_embeddings_dir) if package_description_embeddings_dir else None
        
        # Storage for loaded data
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.bm25_indexes: Dict[str, bm25s.BM25] = {}
        self.bm25_module_paths: Dict[str, List[str]] = {}
        
        # Package description embeddings for boosting
        self.package_description_embeddings: Dict[str, np.ndarray] = {}
        
        # Module type information for filtering
        self.module_type_info: Dict[str, bool] = {}
        
        # Initialize embedding API
        self.api_url = api_url
        self.embedding_url = f"{api_url}/embedding"
        print(f"Using embedding API at: {self.embedding_url}")
        
        # Test the API
        try:
            response = requests.get(api_url + "/health", timeout=5)
            if response.status_code == 200:
                print("API health check passed")
        except:
            print("API health check failed - continuing anyway")
        
    def discover_available_packages(self) -> List[str]:
        """Discover all packages with embeddings available."""
        packages_dir = self.embedding_dir / "packages"
        if not packages_dir.exists():
            return []
            
        packages = []
        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir():
                # Check if it has the required files
                embedding_path = package_dir / "embeddings.npz"
                metadata_path = package_dir / "metadata.json"
                if embedding_path.exists() and metadata_path.exists():
                    packages.append(package_dir.name)
                    
        return sorted(packages)
        
    def load_module_type_info(self, package_names: List[str]) -> None:
        """Load module type information for the specified packages."""
        descriptions_dir = Path("module-descriptions")
        if not descriptions_dir.exists():
            print("Warning: module-descriptions directory not found")
            return
            
        loaded_count = 0
        module_type_count = 0
        
        for package in package_names:
            json_file = descriptions_dir / f"{package}.json"
            if not json_file.exists():
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract module type info from each library
                for library_data in data.get('libraries', {}).values():
                    for module_path, module_info in library_data.get('modules', {}).items():
                        if isinstance(module_info, dict) and 'is_module_type' in module_info:
                            is_module_type = module_info['is_module_type']
                            self.module_type_info[module_path] = is_module_type
                            loaded_count += 1
                            if is_module_type:
                                module_type_count += 1
                        elif isinstance(module_info, str):
                            # Old format - assume not a module type
                            self.module_type_info[module_path] = False
                            loaded_count += 1
                    
            except Exception as e:
                print(f"Warning: Failed to load module type info from {json_file}: {e}")
                
        print(f"Loaded module type info for {loaded_count} modules ({module_type_count} module types)")
    
    def load_package_description_embeddings(self, package_names: List[str]) -> None:
        """Load pre-computed package description embeddings for the specified packages."""
        if not self.package_description_embeddings_dir or not self.package_description_embeddings_dir.exists():
            print("Warning: package-description-embeddings directory not found")
            return
            
        packages_dir = self.package_description_embeddings_dir / "packages"
        if not packages_dir.exists():
            print("Warning: package-description-embeddings/packages directory not found")
            return
            
        loaded_count = 0
        
        for package in package_names:
            package_dir = packages_dir / package
            if not package_dir.exists():
                continue
                
            try:
                # Load embedding
                embeddings_file = package_dir / "embeddings.npz"
                if embeddings_file.exists():
                    data = np.load(embeddings_file)
                    # Package embeddings have shape (1, embedding_dim)
                    package_embedding = data['embeddings'][0]
                    self.package_description_embeddings[package] = package_embedding
                    loaded_count += 1
                    
            except Exception as e:
                print(f"Warning: Failed to load package description embedding for {package}: {e}")
                
        print(f"Loaded {loaded_count} package description embeddings")
    
    def load_package_data(self, package_names: Optional[List[str]] = None) -> None:
        """Load embeddings and BM25 indexes for specified packages or all available."""
        if package_names is None:
            # Load all available packages
            package_names = self.discover_available_packages()
            print(f"Found {len(package_names)} packages with embeddings")
            if len(package_names) > 100:
                # For large numbers, show progress differently
                print("Loading all packages (this may take a moment)...")
        else:
            print(f"Loading data for {len(package_names)} specified packages...")
        
        # Use tqdm only if reasonable number of packages
        iterator = tqdm(package_names, desc="Loading packages") if len(package_names) <= 100 else package_names
        
        loaded_embeddings = 0
        loaded_indexes = 0
        
        for i, package in enumerate(iterator):
            if len(package_names) > 100 and i % 100 == 0:
                print(f"  Loaded {i}/{len(package_names)} packages...")
                
            # Load embeddings
            embedding_path = self.embedding_dir / "packages" / package / "embeddings.npz"
            metadata_path = self.embedding_dir / "packages" / package / "metadata.json"
            
            if embedding_path.exists() and metadata_path.exists():
                try:
                    # Load embeddings
                    data = np.load(embedding_path)
                    package_embeddings = data['embeddings']
                    
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        package_metadata = json.load(f)
                    
                    # Store with package prefix in module paths
                    for i, module_info in enumerate(package_metadata['modules']):
                        full_path = f"{package}::{module_info['module_path']}"
                        self.embeddings[full_path] = package_embeddings[i]
                        metadata_entry = {
                            'package': package,
                            'module_path': module_info['module_path'],
                            'description': module_info.get('description', '')
                        }
                        # Include library information if available
                        if 'library' in module_info:
                            metadata_entry['library'] = module_info['library']
                        self.metadata[full_path] = metadata_entry
                    
                    loaded_embeddings += 1
                except Exception as e:
                    print(f"Warning: Failed to load embeddings for {package}: {e}")
            
            # Load BM25 index
            index_path = self.index_dir / package
            if index_path.exists():
                try:
                    # Load the BM25 index
                    retriever = bm25s.BM25.load(str(index_path / "index"))
                    self.bm25_indexes[package] = retriever
                    
                    # Load module paths for this package
                    with open(index_path / "module_paths.json", 'r') as f:
                        self.bm25_module_paths[package] = json.load(f)
                    
                    loaded_indexes += 1
                except Exception as e:
                    print(f"Warning: Failed to load BM25 index for {package}: {e}")
                    
        print(f"\nLoaded {loaded_embeddings} package embeddings and {loaded_indexes} BM25 indexes")
        print(f"Total modules: {len(self.embeddings)}")
        
        # Load module type information for filtering
        if package_names:
            self.load_module_type_info(package_names)
            
        # Load package description embeddings for boosting
        if package_names and self.package_description_embeddings_dir:
            self.load_package_description_embeddings(package_names)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query using llama-server API."""
        try:
            payload = {"content": query}
            
            response = requests.post(
                self.embedding_url,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text[:200]}")
                
            result = response.json()
            
            # Handle llama-server response format
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'embedding' in result[0]:
                    # Extract the embedding (it's nested)
                    embedding_data = result[0]['embedding']
                    if isinstance(embedding_data, list) and len(embedding_data) > 0:
                        embedding = np.array(embedding_data[0], dtype=np.float32)
                    else:
                        embedding = np.array(embedding_data, dtype=np.float32)
                else:
                    raise Exception(f"Unexpected response format: {result[:200]}")
            else:
                raise Exception(f"Unexpected response format: {result[:200]}")
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            print(f"Failed to generate embedding: {e}")
            raise
    
    def semantic_search(self, query: str, top_k: int = 10, package_boost_weight: float = 0.2) -> List[Tuple[str, float]]:
        """Search using semantic similarity with package description boosting."""
        query_embedding = self.embed_query(query)
        
        # Calculate base similarities
        similarities = []
        package_names = []
        
        for module_path, module_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, module_embedding)
            similarities.append(similarity)
            
            # Extract package name from module path (format: "package::module.path")
            if "::" in module_path:
                package_name = module_path.split("::")[0]
            else:
                package_name = "unknown"
            package_names.append(package_name)
        
        # Calculate package boosts if available
        package_boosts = {}
        if self.package_description_embeddings and package_boost_weight > 0:
            package_boosts = self.calculate_package_boost(query_embedding, package_names)
        
        # Combine scores
        combined_similarities = []
        module_paths = list(self.embeddings.keys())
        
        for i, (module_path, base_sim, package_name) in enumerate(zip(module_paths, similarities, package_names)):
            if package_boosts:
                package_boost = package_boosts.get(package_name, 0.0)
                # Weighted combination
                combined_score = (1 - package_boost_weight) * base_sim + package_boost_weight * package_boost
            else:
                combined_score = base_sim
                
            combined_similarities.append((module_path, float(combined_score)))
        
        # Sort by combined similarity
        combined_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return combined_similarities[:top_k]
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 keyword matching."""
        all_results = []
        
        # Search in each package's index
        for package, retriever in self.bm25_indexes.items():
            # Tokenize query with progress suppressed
            import contextlib
            import io
            with contextlib.redirect_stderr(io.StringIO()):
                query_tokens = bm25s.tokenize(query, show_progress=False)
            
            # Get number of documents in this index
            module_paths = self.bm25_module_paths[package]
            k = min(top_k, len(module_paths))
            
            if k == 0:
                continue
                
            # Search with progress suppressed
            with contextlib.redirect_stderr(io.StringIO()):
                results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
            
            # Map indices to module paths
            for i, (doc_indices, doc_scores) in enumerate(zip(results, scores)):
                for idx, score in zip(doc_indices, doc_scores):
                    if idx < len(module_paths):
                        # Convert module path array to string
                        if isinstance(module_paths[idx], list):
                            module_path_str = "::".join(module_paths[idx])
                        else:
                            module_path_str = str(module_paths[idx])
                        full_path = f"{package}::{module_path_str}"
                        all_results.append((full_path, float(score)))
        
        # Sort all results by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results[:top_k]
    
    def calculate_package_boost(self, query_embedding: np.ndarray, package_names: List[str]) -> Dict[str, float]:
        """Calculate package description similarity boost using pre-computed embeddings."""
        if not self.package_description_embeddings:
            return {}
        
        # Only calculate for unique packages that have embeddings
        unique_packages = set(package_names)
        
        package_boosts = {}
        boost_count = 0
        
        for package in unique_packages:
            if package in self.package_description_embeddings:
                # Use pre-computed package embedding
                package_embedding = self.package_description_embeddings[package]
                
                # Calculate similarity
                similarity = float(np.dot(query_embedding, package_embedding))
                package_boosts[package] = similarity
                boost_count += 1
            else:
                package_boosts[package] = 0.0
        
        print(f"Calculated package boosts for {boost_count} packages (using pre-computed embeddings)")
        return package_boosts
    
    def search(self, query: str, top_k: int = 10, package_boost_weight: float = 0.2) -> Dict[str, List[Dict]]:
        """Perform unified search combining semantic and keyword approaches."""
        # Get more results initially to account for filtering (estimate ~10% module types)
        initial_k = max(top_k * 2, 20)
        semantic_results = self.semantic_search(query, initial_k, package_boost_weight)
        keyword_results = self.keyword_search(query, initial_k)
        
        # Convert to detailed format, filtering out module types
        def format_results(results: List[Tuple[str, float]]) -> List[Dict]:
            formatted = []
            for module_path, score in results:
                if module_path in self.metadata:
                    meta = self.metadata[module_path]
                    raw_module_path = meta['module_path']
                    
                    # Skip module types
                    is_module_type = self.module_type_info.get(raw_module_path, False)
                    if is_module_type:
                        continue
                        
                    formatted.append({
                        'package': meta['package'],
                        'module_path': raw_module_path,
                        'description': meta['description'][:150] + '...' if len(meta['description']) > 150 else meta['description'],
                        'score': score,
                        'library': meta.get('library', '')
                    })
                    
                    # Stop when we have enough results
                    if len(formatted) >= top_k:
                        break
            return formatted
        
        return {
            'semantic': format_results(semantic_results),
            'keyword': format_results(keyword_results)
        }


def print_results(results: Dict[str, List[Dict]], query: str, packages_info: str):
    """Pretty print search results."""
    print(f"\n🔍 Search results for: '{query}'")
    print(f"📦 {packages_info}")
    
    print("\n🧠 Semantic Search Results (Embeddings):")
    print("─" * 80)
    for i, result in enumerate(results['semantic'], 1):
        print(f"{i}. {result['package']}::{result['module_path']} (score: {result['score']:.4f})")
        if result.get('library'):
            print(f"   Library: {result['library']}")
        print(f"   {result['description']}")
        print()
    
    print("\n📝 Full-Text Search Results (BM25):")
    print("─" * 80)
    for i, result in enumerate(results['keyword'], 1):
        # For BM25 results, show module path in a more readable format
        module_path = result['module_path']
        if module_path.startswith('[') and module_path.endswith(']'):
            # Parse the list representation
            try:
                path_parts = eval(module_path)
                module_path = '::'.join(path_parts)
            except:
                pass
        
        print(f"{i}. {result['package']}::{module_path} (score: {result['score']:.4f})")
        if result.get('library'):
            print(f"   Library: {result['library']}")
        if result['description']:
            print(f"   {result['description']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Unified search for OCaml modules")
    parser.add_argument("query", help="Search query")
    parser.add_argument("packages", nargs="*", help="Packages to search in (if empty, searches all)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--embedding-dir", default="package-embeddings",
                        help="Directory containing package embeddings")
    parser.add_argument("--index-dir", default="module-indexes",
                        help="Directory containing BM25 indexes")
    parser.add_argument("--api-url", default="http://localhost:8080",
                        help="llama-server API URL")
    parser.add_argument("--format", choices=['text', 'json'], default='text',
                        help="Output format")
    parser.add_argument("--no-popularity", action="store_true",
                        help="Disable popularity weighting")
    parser.add_argument("--popularity-weight", type=float, default=0.3,
                        help="Weight for popularity scoring")
    
    args = parser.parse_args()
    
    # Initialize search engine
    engine = UnifiedSearchEngine(args.embedding_dir, args.index_dir, args.api_url)
    
    # Load package data
    if args.packages:
        engine.load_package_data(args.packages)
        packages_info = f"Packages searched: {', '.join(args.packages)}"
    else:
        engine.load_package_data(None)  # Load all
        packages_info = f"Searched across all {len(engine.discover_available_packages())} available packages"
    
    # Perform search
    results = engine.search(args.query, args.top_k)
    
    # Output results
    if args.format == 'json':
        print(json.dumps(results, indent=2))
    else:
        print_results(results, args.query, packages_info)


if __name__ == "__main__":
    main()