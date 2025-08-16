#!/usr/bin/env python3
"""
Generate embeddings for OCaml package descriptions.

This script reads package descriptions from package-descriptions/ and generates
embeddings using a local embedding model server.
"""

import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False

def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

@dataclass
class PackageDescription:
    """Package description data."""
    name: str
    version: str
    description: str

@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    package_name: str
    embedding: np.ndarray
    success: bool
    error: Optional[str] = None

class EmbeddingClient:
    """Client for generating embeddings using a local embedding server."""
    
    def __init__(self, base_url: str = "http://localhost:8080", model: str = "Qwen/Qwen3-Embedding-8B-GGUF"):
        self.base_url = base_url
        self.model = model
        self.embedding_url = f"{base_url}/embedding"
        print(f"EmbeddingClient initialized with base_url={base_url}, model={model}")
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a text string."""
        try:
            payload = {
                "model": self.model,
                "input": text
            }
            
            response = requests.post(
                self.embedding_url,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"API returned status {response.status_code}: {response.text[:200]}")
                return None
                
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                # Check if it's a list of dicts (OpenAI-style)
                if isinstance(result[0], dict) and 'embedding' in result[0]:
                    raw_embedding = np.array(result[0]['embedding'], dtype=np.float32)
                elif isinstance(result[0], (list, np.ndarray)):
                    # Response is a list of embeddings
                    raw_embedding = np.array(result[0], dtype=np.float32)
                else:
                    return None
            elif isinstance(result, dict) and 'data' in result and len(result['data']) > 0:
                # OpenAI-style response format
                raw_embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)
            else:
                return None
                
            # Handle multi-dimensional embeddings by taking the mean
            if raw_embedding.ndim == 2:
                # If we have a 2D array (sequence_length, embedding_dim), take the mean
                embedding = np.mean(raw_embedding, axis=0)
            elif raw_embedding.ndim == 1:
                # If we have a 1D array, use it directly
                embedding = raw_embedding
            else:
                return None
                
            # Validate embedding
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                return None
                
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
                
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

def load_package_descriptions(input_dir: Path) -> List[PackageDescription]:
    """Load all package descriptions from JSON files."""
    descriptions = []
    json_files = list(input_dir.glob("*.json"))
    
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    
    with tqdm(json_files, desc="Loading package descriptions", unit="file") as pbar:
        for json_file in pbar:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'package' in data and 'description' in data:
                    descriptions.append(PackageDescription(
                        name=data['package'],
                        version=data.get('version', 'unknown'),
                        description=data['description']
                    ))
                    pbar.set_postfix({"loaded": len(descriptions)})
                else:
                    logger.warning(f"Invalid format in {json_file}")
                    
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Successfully loaded {len(descriptions)} package descriptions")
    return descriptions

def generate_package_embedding(package: PackageDescription, client: EmbeddingClient) -> EmbeddingResult:
    """Generate embedding for a single package description."""
    global shutdown_requested
    
    if shutdown_requested:
        return EmbeddingResult(package.name, None, False, "Shutdown requested")
    
    try:
        # Create embedding text that includes package name and description
        embedding_text = f"Package: {package.name}\nDescription: {package.description}"
        
        embedding = client.generate_embedding(embedding_text)
        
        if embedding is not None:
            return EmbeddingResult(package.name, embedding, True)
        else:
            return EmbeddingResult(package.name, None, False, "Failed to generate embedding")
            
    except Exception as e:
        return EmbeddingResult(package.name, None, False, f"Exception: {str(e)}")

def process_packages_batch(packages: List[PackageDescription], client: EmbeddingClient, 
                          batch_size: int = 16) -> List[EmbeddingResult]:
    """Process a batch of packages with rate limiting and progress reporting."""
    results = []
    total_batches = (len(packages) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(packages)} packages in {total_batches} batches of {batch_size}")
    
    with tqdm(total=len(packages), desc="Generating embeddings", unit="pkg") as pbar:
        for i in range(0, len(packages), batch_size):
            if shutdown_requested:
                logger.info("Shutdown requested, stopping processing")
                break
                
            batch = packages[i:i + batch_size]
            batch_results = []
            batch_num = i // batch_size + 1
            
            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} packages)")
            
            # Process batch with some parallelization
            with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
                future_to_package = {
                    executor.submit(generate_package_embedding, package, client): package
                    for package in batch
                }
                
                for future in as_completed(future_to_package):
                    if shutdown_requested:
                        break
                    result = future.result()
                    batch_results.append(result)
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Update status in progress bar
                    successful = sum(1 for r in results + batch_results if r.success)
                    total_processed = len(results) + len(batch_results)
                    pbar.set_postfix({
                        "success": f"{successful}/{total_processed}",
                        "batch": f"{batch_num}/{total_batches}"
                    })
            
            results.extend(batch_results)
            
            # Rate limiting between batches
            if i + batch_size < len(packages) and not shutdown_requested:
                time.sleep(0.1)  # Small delay between batches
    
    return results

def save_embeddings(results: List[EmbeddingResult], output_dir: Path) -> None:
    """Save embeddings in per-package structure like module embeddings."""
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        logger.warning("No successful embeddings to save")
        return
    
    logger.info(f"Saving {len(successful_results)} successful embeddings")
    
    # Check embedding dimensions
    embedding_dims = [r.embedding.shape for r in successful_results]
    unique_dims = set(embedding_dims)
    
    if len(unique_dims) > 1:
        logger.error(f"Embeddings have inconsistent dimensions: {unique_dims}")
        # Find which packages have different dimensions
        for i, (result, dim) in enumerate(zip(successful_results[:10], embedding_dims[:10])):
            logger.error(f"  {result.package_name}: {dim}")
        return
    
    # Create packages directory
    packages_dir = output_dir / "packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each package separately with progress bar
    with tqdm(successful_results, desc="Saving embeddings", unit="pkg") as pbar:
        for result in pbar:
            pbar.set_postfix({"current": result.package_name})
            
            package_dir = packages_dir / result.package_name
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # Save embedding
            embeddings_file = package_dir / "embeddings.npz"
            np.savez_compressed(embeddings_file, embeddings=result.embedding[None, :])  # Add batch dimension
            
            # Save package metadata
            metadata = {
                "package": result.package_name,
                "embedding_dimension": result.embedding.shape[0],
                "generation_time": time.time()
            }
            
            metadata_file = package_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
    
    # Save global metadata
    global_metadata = {
        "total_packages": len(successful_results),
        "embedding_dimension": successful_results[0].embedding.shape[0],
        "packages": [r.package_name for r in successful_results],
        "generation_time": time.time()
    }
    
    global_metadata_file = output_dir / "metadata.json"
    with open(global_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f, indent=2)
    
    logger.info(f"Saved {len(successful_results)} package embeddings to {packages_dir}")
    logger.info(f"Saved global metadata to {global_metadata_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for OCaml package descriptions")
    parser.add_argument("--input-dir", default="package-descriptions", 
                       help="Directory containing package description JSON files")
    parser.add_argument("--output-dir", default="package-description-embeddings", 
                       help="Output directory for package description embeddings")
    parser.add_argument("--embedding-url", default="http://localhost:8080",
                       help="Base URL for embedding server")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                       help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of packages to process (for testing)")
    parser.add_argument("--packages",
                       help="Comma-separated list of specific packages to process")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load package descriptions
    logger.info(f"Loading package descriptions from {input_dir}")
    packages = load_package_descriptions(input_dir)
    
    if not packages:
        logger.error("No package descriptions found")
        return 1
    
    logger.info(f"Found {len(packages)} package descriptions")
    
    # Filter by specific packages if specified
    if args.packages:
        specified_packages = set(args.packages.split(","))
        original_count = len(packages)
        packages = [p for p in packages if p.name in specified_packages]
        logger.info(f"Filtered from {original_count} to {len(packages)} packages: {specified_packages}")
    
    # Apply limit if specified
    if args.limit:
        original_count = len(packages)
        packages = packages[:args.limit]
        logger.info(f"Limited from {original_count} to {len(packages)} packages")
    
    if not packages:
        logger.warning("No packages to process after filtering")
        return 0
    
    # Initialize embedding client
    try:
        logger.info(f"Initializing embedding client: {args.embedding_url} with model {args.model}")
        client = EmbeddingClient(args.embedding_url, args.model)
    except Exception as e:
        logger.error(f"Failed to initialize embedding client: {e}")
        return 1
    
    # Generate embeddings
    logger.info(f"Starting embedding generation for {len(packages)} packages")
    logger.info(f"Configuration: batch_size={args.batch_size}, model={args.model}")
    start_time = time.time()
    
    results = process_packages_batch(packages, client, args.batch_size)
    
    if shutdown_requested:
        logger.warning("Processing interrupted by shutdown signal")
        return 1
    
    # Report results
    successful = len([r for r in results if r.success])
    failed = len([r for r in results if not r.success])
    
    elapsed = time.time() - start_time
    rate = len(packages) / elapsed if elapsed > 0 else 0
    
    logger.info(f"Embedding generation complete in {elapsed:.1f}s ({rate:.1f} packages/sec)")
    logger.info(f"Results: {successful}/{len(packages)} successful, {failed} failed")
    
    if failed > 0:
        logger.warning(f"Failed to process {failed} packages:")
        # Log first few failures for debugging
        failed_results = [r for r in results if not r.success][:5]
        for result in failed_results:
            logger.warning(f"  {result.package_name}: {result.error}")
        if len(failed_results) < failed:
            logger.warning(f"  ... and {failed - len(failed_results)} more failures")
    
    # Calculate success rate and stats
    success_rate = (successful / len(packages)) * 100 if packages else 0
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Save embeddings
    if successful > 0:
        logger.info("Saving embeddings to disk...")
        save_embeddings(results, output_dir)
        logger.info(f"Package embeddings successfully saved to {output_dir}")
        
        # Final summary
        logger.info("=== SUMMARY ===")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Packages processed: {len(packages)}")
        logger.info(f"Successful embeddings: {successful}")
        logger.info(f"Failed embeddings: {failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Processing rate: {rate:.1f} packages/sec")
    else:
        logger.error("No successful embeddings to save")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())