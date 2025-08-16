#!/usr/bin/env python3
"""
FastMCP Server for OCaml Documentation Search

This server exposes tools for searching and interacting with the OCaml
documentation dataset through the Model Context Protocol (MCP) using HTTP SSE.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

# Import semantic search components
from semantic_search_fixed import SemanticSearchEngine
from unified_search_all import UnifiedSearchEngine
from semantic_search import SemanticSearch
from version_utils import find_latest_version

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ocaml-search", host="0.0.0.0")

# Global search engine instances (lazy loaded)  
unified_searcher: Optional[UnifiedSearchEngine] = None
semantic_searcher: Optional[SemanticSearch] = None
embeddings_dir = Path("package-embeddings")
package_descriptions_dir = Path("package-descriptions")
package_description_embeddings_dir = Path("package-description-embeddings")
indexes_dir = Path("module-indexes")
module_descriptions_dir = Path("module-descriptions")

@mcp.tool()
async def get_package_summary(package_name: str) -> Dict[str, Any]:
    """
    Get a concise overview of what an OCaml package does.
    
    Provides a 3-4 sentence summary explaining the package's purpose, main features,
    and typical use cases. Helpful for understanding a package before using it.
    
    Args:
        package_name: The OCaml package name you want to learn about:
                     - 'lwt' (asynchronous programming)
                     - 'base' (alternative standard library)  
                     - 'cohttp' (HTTP client/server)
                     - 'cmdliner' (command-line interfaces)
                     - 'yojson' (JSON processing)
    
    Returns:
        Package name, version, and description explaining what it does
    """
    try:
        description_file = package_descriptions_dir / f"{package_name}.json"
        
        if not description_file.exists():
            return {
                "error": f"No description found for package '{package_name}'. Package may not exist or description not yet generated."
            }
        
        with open(description_file, 'r', encoding='utf-8') as f:
            package_data = json.load(f)
        
        return {
            "package": package_data["package"],
            "version": package_data["version"],
            "description": package_data["description"]
        }
        
    except Exception as e:
        error_msg = f"Failed to get package summary: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def search_ocaml_packages(query: str, top_k: int = 10, use_popularity_ranking: bool = True, 
                               popularity_weight: float = 0.3) -> Dict[str, Any]:
    """
    Search for OCaml packages based on their descriptions and purposes.
    
    This tool searches across package-level descriptions to find packages that provide
    the functionality you're looking for, rather than specific modules within packages.
    Useful for discovering new packages or finding alternatives.
    
    Args:
        query: What kind of functionality you're looking for in a package:
               - "JSON processing library"
               - "HTTP server framework" 
               - "async programming library"
               - "command line argument parsing"
               - "testing framework"
               - "database bindings"
               - "cryptographic functions"
        top_k: Maximum number of packages to return (default: 10)
        use_popularity_ranking: Whether to factor in package popularity (reverse dependencies) 
                               when ranking results (default: True)
        popularity_weight: How much to weight popularity vs semantic similarity (0.0-1.0, default: 0.3)
                          Higher values favor more popular packages
    
    Returns:
        List of matching packages with:
        - package: Package name
        - version: Package version  
        - description: What the package does
        - similarity_score: How well it matches your query
        - reverse_dependencies: Number of other packages that depend on this one (popularity indicator)
        - combined_score: Final ranking score (only present when use_popularity_ranking=True)
    """
    global semantic_searcher
    
    # Initialize semantic search engine on first use (lazy loading)
    if semantic_searcher is None:
        logger.info("Initializing semantic search engine for package search...")
        try:
            semantic_searcher = SemanticSearch(
                embeddings_dir=embeddings_dir,
                model_name="Qwen/Qwen3-Embedding-8B",  # Use 8B model to match package embeddings
                package_descriptions_dir=package_descriptions_dir,
                package_description_embeddings_dir=package_description_embeddings_dir,
                server_url="http://localhost:8080"
            )
        except Exception as e:
            error_msg = f"Failed to initialize semantic search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Perform package-only search
        results = semantic_searcher.search_packages_only(
            query=query,
            top_k=top_k,
            use_popularity_ranking=use_popularity_ranking,
            popularity_weight=popularity_weight
        )
        
        # Format results for MCP response
        formatted_results = []
        for result in results:
            formatted_result = {
                "package": result["package"],
                "version": result["version"],
                "description": result["description"],
                "similarity_score": round(result["similarity_score"], 4),
                "reverse_dependencies": result["reverse_dependencies"],
                "rank": result["rank"]
            }
            
            # Add combined score if using popularity ranking
            if use_popularity_ranking and "combined_score" in result:
                formatted_result["combined_score"] = round(result["combined_score"], 4)
            
            formatted_results.append(formatted_result)
        
        return {
            "query": query,
            "search_type": "package_descriptions",
            "use_popularity_ranking": use_popularity_ranking,
            "popularity_weight": popularity_weight if use_popularity_ranking else None,
            "total_results": len(formatted_results),
            "packages": formatted_results
        }
        
    except Exception as e:
        error_msg = f"Package search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def search_ocaml(query: str, packages: Optional[List[str]] = None, top_k: int = 80, 
                      popularity_weight: float = 0.3) -> Dict[str, Any]:
    """
    Search for OCaml modules across the entire ecosystem.
    
    Searches all OCaml packages by default, or specific packages if provided.
    Uses both semantic understanding and keyword matching for comprehensive results.
    
    Args:
        query: What you're looking for. Be specific:
               - "MD5 hash function" 
               - "HTTP client for making requests"
               - "JSON parsing and serialization"
               - "list sorting operations"
               - "TCP socket server"
               - "date time parsing and formatting"
        packages: Optional list of specific packages to search within. If not provided,
                 searches all available packages. Common packages include:
                 ['base', 'core', 'lwt', 'async', 'cohttp', 'yojson', 'cmdliner']
        top_k: Maximum number of results to return (default: 8)
        popularity_weight: Weight for popularity in ranking (0.0-1.0, default: 0.3)
    
    Returns:
        Two lists of matching modules:
        - semantic_results: Modules with conceptually similar functionality (includes descriptions)
        - keyword_results: Modules containing your exact keywords in their documentation
        
        Each result includes the package name, module name, and module path.
    """
    global unified_searcher
    
    # Initialize unified search engine on first use (lazy loading)
    if unified_searcher is None:
        logger.info("Initializing unified search engine...")
        try:
            unified_searcher = UnifiedSearchEngine(
                embedding_dir=embeddings_dir,
                index_dir=indexes_dir,
                api_url="http://localhost:8080",
                package_description_embeddings_dir=package_description_embeddings_dir
            )
        except Exception as e:
            error_msg = f"Failed to initialize unified search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Load specified packages and perform search
        if packages:
            unified_searcher.load_package_data(packages)
        else:
            unified_searcher.load_package_data(None)  # Load all packages
            
        results = unified_searcher.search(query, top_k)
        
        # Format results for MCP response
        return {
            "query": query,
            "packages_searched": packages if packages else "all",
            "semantic_results": [
                {
                    "package": r["package"],
                    "library": r.get("library", ""),
                    "module": r["module_path"],
                    "description": r["description"],
                    "score": r["score"]
                }
                for r in results["semantic"]
            ],
            "keyword_results": [
                {
                    "package": r["package"],
                    "library": r.get("library", ""),
                    "module": r["module_path"],
                    "description": r["description"],
                    "score": r["score"]
                }
                for r in results["keyword"]
            ],
            "total_results": len(results["semantic"]) + len(results["keyword"])
        }
        
    except Exception as e:
        error_msg = f"Unified search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def build_module_location_map() -> Dict[str, List[Dict[str, str]]]:
    """Build a map of module names to their locations (package/library/version).
    
    Returns two maps:
    - full_map: Maps full module paths (e.g., "Stdlib-List") to locations
    - basename_map: Maps base names (e.g., "List") to all modules ending with that name
    """
    full_map = {}
    basename_map = {}
    docs_dir = Path("docs-md")
    
    if not docs_dir.exists():
        return full_map, basename_map
    
    for package_dir in docs_dir.iterdir():
        if not package_dir.is_dir() or package_dir.name.startswith('.'):
            continue
            
        package_name = package_dir.name
        
        # Find latest version for this package
        available_versions = []
        for version_dir in package_dir.iterdir():
            if version_dir.is_dir() and not version_dir.name.startswith('.'):
                available_versions.append(version_dir.name)
        
        if not available_versions:
            continue
            
        latest_version, _ = find_latest_version(available_versions)
        if not latest_version:
            continue
            
        # Check doc directory structure
        doc_dir = package_dir / latest_version / "doc"
        if not doc_dir.exists():
            continue
            
        # Scan all libraries in this package version
        for library_dir in doc_dir.iterdir():
            if not library_dir.is_dir() or library_dir.name.startswith('.'):
                continue
            if library_dir.name in ['src', 'LICENSE.md', 'README.md', 'index.md', 'manual.md']:
                continue
                
            library_name = library_dir.name
            
            # Scan all module files in this library
            for module_file in library_dir.glob("*.md"):
                if module_file.name in ['index.md', 'LICENSE.md', 'README.md', 'manual.md']:
                    continue
                    
                module_name = module_file.stem  # Remove .md extension
                
                location = {
                    "package": package_name,
                    "library": library_name,
                    "version": latest_version,
                    "file_path": str(module_file),
                    "full_module_path": module_name
                }
                
                # Add to full map
                if module_name not in full_map:
                    full_map[module_name] = []
                full_map[module_name].append(location)
                
                # Also add to basename map for partial matching
                # Extract the last component (e.g., "List" from "Stdlib-List")
                # In OCaml docs, submodules are often represented with hyphens
                parts = module_name.split('-')
                basename = parts[-1] if parts else module_name
                
                if basename not in basename_map:
                    basename_map[basename] = []
                basename_map[basename].append(location)
    
    return full_map, basename_map


@mcp.tool()
async def get_module_docs(module_path: str, package: Optional[str] = None, library: Optional[str] = None, version: Optional[str] = None) -> Dict[str, Any]:
    """
    Get raw documentation content for a specific OCaml module.
    
    Returns the raw markdown documentation for a module from the docs-md directory.
    If package/library are not specified, searches all packages/libraries for the module.
    If multiple locations are found, returns information about all available locations.
    
    Args:
        module_path: The module path with dots replaced by hyphens (e.g., "Gg-V3", "Lwt-Syntax", "List")
        package: The OCaml package name (e.g., "gg", "lwt", "base"). If not specified, searches all packages.
        library: The library name within the package (e.g., "gg", "lwt.unix"). If not specified, searches all libraries.
        version: The package version (e.g., "1.0.0", "5.9.1"). If not specified, uses the latest available version.
    
    Returns:
        Dictionary containing either:
        - Single module result: module_info, content, file_path
        - Multiple locations: available_locations list if module found in multiple places
        - Error: error message if module not found or cannot be read
    """
    try:
        # Convert dots to hyphens for file system representation
        module_path_for_fs = module_path.replace('.', '-')
        
        # If both package and library are specified, use the original direct approach
        if package and library:
            return await _get_module_docs_direct(module_path_for_fs, package, library, version)
        
        # If package is specified but library is not, search within that package
        if package and not library:
            return await _get_module_docs_in_package(module_path, package, version)
        
        # If neither package nor library is specified, search everywhere
        logger.info(f"Searching for module '{module_path}' across all packages...")
        full_map, basename_map = build_module_location_map()
        
        # First, check if the user provided a hierarchical path with dots (e.g., "Stdlib.List")
        # Convert dots to hyphens for the file system representation
        module_path_hyphenated = module_path.replace('.', '-')
        
        # Try exact match first (full module path)
        exact_matches = []
        if module_path_hyphenated in full_map:
            exact_matches = full_map[module_path_hyphenated]
            logger.info(f"Found exact match for '{module_path}' ({module_path_hyphenated})")
        
        # If no exact match, try basename matching (partial match)
        partial_matches = []
        if not exact_matches:
            # Extract the last component for basename search
            basename = module_path.split('.')[-1]
            basename_hyphenated = basename.replace('.', '-')
            
            if basename_hyphenated in basename_map:
                partial_matches = basename_map[basename_hyphenated]
                logger.info(f"Found partial matches for basename '{basename}'")
        
        # Combine results, preferring exact matches
        locations = exact_matches if exact_matches else partial_matches
        
        if not locations:
            return {
                "error": f"Module '{module_path}' not found in any package",
                "searched_module": module_path,
                "search_type": "exact and partial"
            }
        
        # If only one location found, return the content directly
        if len(locations) == 1:
            location = locations[0]
            actual_module_path = location['full_module_path']
            logger.info(f"Found unique module '{actual_module_path}' in {location['package']}.{location['library']}")
            return await _get_module_docs_direct(
                actual_module_path, 
                location['package'], 
                location['library'], 
                version or location['version']
            )
        
        # Multiple locations found - return information about all locations
        search_type = "exact match" if exact_matches else "partial match (basename)"
        logger.info(f"Found module '{module_path}' in {len(locations)} locations ({search_type})")
        
        return {
            "searched_module": module_path,
            "search_type": search_type,
            "found_in_multiple_locations": True,
            "available_locations": [
                {
                    "package": loc["package"],
                    "library": loc["library"],
                    "version": loc["version"],
                    "full_module_path": loc["full_module_path"]
                }
                for loc in locations
            ],
            "total_locations": len(locations),
            "suggestion": f"Please specify package and/or library to get the documentation. For example: get_module_docs('{locations[0]['full_module_path']}', package='{locations[0]['package']}', library='{locations[0]['library']}')"
        }
        
    except Exception as e:
        error_msg = f"Failed to search for module documentation: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "searched_module": module_path
        }


async def _get_module_docs_direct(module_path: str, package: str, library: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Get module docs when package and library are both specified."""
    try:
        docs_dir = Path("docs-md")
        package_dir = docs_dir / package
        
        # Check if package directory exists
        if not package_dir.exists():
            return {
                "error": f"Package not found: {package}",
                "module_info": {
                    "package": package,
                    "version": version,
                    "library": library,
                    "module_path": module_path
                }
            }
        
        # If version not specified, find the latest version
        if version is None:
            available_versions = []
            for item in package_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    available_versions.append(item.name)
            
            if not available_versions:
                return {
                    "error": f"No versions found for package: {package}",
                    "module_info": {
                        "package": package,
                        "version": None,
                        "library": library,
                        "module_path": module_path
                    }
                }
            
            # Find the latest version using opam version ordering
            latest_version, _ = find_latest_version(available_versions)
            if latest_version is None:
                return {
                    "error": f"Could not determine latest version for package: {package}",
                    "module_info": {
                        "package": package,
                        "version": None,
                        "library": library,
                        "module_path": module_path
                    }
                }
            version = latest_version
            logger.info(f"Using latest version {version} for package {package}")
        
        # Construct the file path: docs-md/<package>/<version>/doc/<library>/<module-path>.md
        doc_file = docs_dir / package / version / "doc" / library / f"{module_path}.md"
        
        # Check if file exists
        if not doc_file.exists():
            return {
                "error": f"Documentation file not found: {doc_file}",
                "module_info": {
                    "package": package,
                    "version": version,
                    "library": library,
                    "module_path": module_path
                },
                "file_path": str(doc_file)
            }
        
        # Read the documentation content
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(doc_file, 'r', encoding='latin-1') as f:
                content = f.read()
        
        return {
            "module_info": {
                "package": package,
                "version": version,
                "library": library,
                "module_path": module_path
            },
            "content": content,
            "file_path": str(doc_file),
            "content_length": len(content)
        }
        
    except Exception as e:
        error_msg = f"Failed to read module documentation: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "module_info": {
                "package": package,
                "version": version,
                "library": library,
                "module_path": module_path
            }
        }


async def _get_module_docs_in_package(module_path: str, package: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Get module docs when package is specified but library is not."""
    try:
        docs_dir = Path("docs-md")
        package_dir = docs_dir / package
        
        # Check if package directory exists
        if not package_dir.exists():
            return {
                "error": f"Package not found: {package}",
                "searched_module": module_path,
                "searched_package": package
            }
        
        # If version not specified, find the latest version
        if version is None:
            available_versions = []
            for item in package_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    available_versions.append(item.name)
            
            if not available_versions:
                return {
                    "error": f"No versions found for package: {package}",
                    "searched_module": module_path,
                    "searched_package": package
                }
            
            # Find the latest version using opam version ordering
            latest_version, _ = find_latest_version(available_versions)
            if latest_version is None:
                return {
                    "error": f"Could not determine latest version for package: {package}",
                    "searched_module": module_path,
                    "searched_package": package
                }
            version = latest_version
            logger.info(f"Using latest version {version} for package {package}")
        
        # Search through all libraries in this package
        doc_dir = package_dir / version / "doc"
        if not doc_dir.exists():
            return {
                "error": f"No documentation directory found for package {package} version {version}",
                "searched_module": module_path,
                "searched_package": package,
                "searched_version": version
            }
        
        # Convert dots to hyphens for file system
        module_path_hyphenated = module_path.replace('.', '-')
        
        found_libraries = []
        for library_dir in doc_dir.iterdir():
            if not library_dir.is_dir() or library_dir.name.startswith('.'):
                continue
            if library_dir.name in ['src', 'LICENSE.md', 'README.md', 'index.md', 'manual.md']:
                continue
                
            library_name = library_dir.name
            
            # Try both the original and hyphenated versions
            for try_path in [module_path_hyphenated, module_path]:
                module_file = library_dir / f"{try_path}.md"
                
                if module_file.exists():
                    found_libraries.append({
                        "library": library_name,
                        "file_path": str(module_file),
                        "actual_module_path": try_path
                    })
                    break
        
        if not found_libraries:
            return {
                "error": f"Module '{module_path}' not found in any library of package '{package}'",
                "searched_module": module_path,
                "searched_package": package,
                "searched_version": version
            }
        
        # If only one library found, return the content directly
        if len(found_libraries) == 1:
            library_info = found_libraries[0]
            actual_path = library_info.get('actual_module_path', module_path)
            logger.info(f"Found module '{module_path}' (as '{actual_path}') in {package}.{library_info['library']}")
            return await _get_module_docs_direct(actual_path, package, library_info['library'], version)
        
        # Multiple libraries found - return information about all libraries
        return {
            "module_path": module_path,
            "package": package,
            "version": version,
            "found_in_multiple_libraries": True,
            "available_libraries": [lib["library"] for lib in found_libraries],
            "total_libraries": len(found_libraries),
            "suggestion": f"Please specify library to get the documentation. Use: get_module_docs('{module_path}', package='{package}', library='LIBRARY_NAME')"
        }
        
    except Exception as e:
        error_msg = f"Failed to search for module in package: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "searched_module": module_path,
            "searched_package": package
        }


@mcp.tool()
async def sherlodoc(query: str) -> Dict[str, Any]:
    """
    Search OCaml documentation using Sherlodoc - particularly effective for type searches.
    
    Sherlodoc is specialized for finding types, functions, and modules across all OCaml packages.
    It excels at type-based queries and can find exact matches for complex type signatures.
    
    Args:
        query: Your search query. Can be:
               - Type signatures: "int -> string -> bool"
               - Module paths: "Base.List.t"
               - Function names: "List.map"
               - Type definitions: "result"
               - Complex types: "('a -> 'b) -> 'a list -> 'b list"
    
    Returns:
        Search results including type definitions, functions, and their documentation
    """
    try:
        # URL encode the query
        encoded_query = quote(query)
        url = f"http://localhost:1234/api?q={encoded_query}"
        
        # Make async HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return {"error": f"Sherlodoc API returned status {response.status}"}
                
                html_content = await response.text()
        
        # Parse HTML response
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract query info
        query_div = soup.find('div', class_='query')
        query_text = query_div.get_text(strip=True) if query_div else f"Results for {query}"
        
        # Extract search results
        results = []
        found_items = soup.find_all('li')
        
        for item in found_items[:20]:  # Limit to first 20 results
            result = {}
            
            # Extract package info
            pkg_div = item.find('div', class_='pkg')
            if pkg_div and pkg_div.find('a'):
                # Note: The HTML seems to have empty package info, we'll work with what we have
                result['package'] = 'Unknown'  # Package info not provided in the HTML
            
            # Extract the main code/signature
            pre_tag = item.find('pre')
            if pre_tag:
                # Get the full text including emphasized parts
                signature_parts = []
                for elem in pre_tag.descendants:
                    if elem.name is None:  # Text node
                        signature_parts.append(str(elem))
                    elif elem.name == 'em':
                        signature_parts.append(elem.get_text())
                
                result['signature'] = ''.join(signature_parts).strip()
                
                # Extract the link if available
                link_tag = pre_tag.find('a')
                if link_tag and link_tag.get('href'):
                    result['url'] = link_tag['href']
                    # Extract module path from the link text
                    em_tag = link_tag.find('em')
                    if em_tag:
                        result['module_path'] = em_tag.get_text()
            
            # Extract documentation/comment
            comment_div = item.find('div', class_='comment')
            if comment_div:
                # Extract text from all paragraphs
                doc_parts = []
                for p in comment_div.find_all('p'):
                    doc_parts.append(p.get_text(strip=True))
                if doc_parts:
                    result['documentation'] = ' '.join(doc_parts)
            
            if result and 'signature' in result:
                results.append(result)
        
        return {
            "query": query,
            "query_info": query_text,
            "results": results,
            "total_results": len(results)
        }
        
    except aiohttp.ClientError as e:
        error_msg = f"Failed to connect to Sherlodoc API: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Sherlodoc search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def main():
    """Main entry point for the FastMCP server."""
    import sys
    import asyncio
    
    # Check for --test flag for direct testing
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run a simple test
        async def test():
            if len(sys.argv) > 2 and sys.argv[2] == "--summary":
                # Test package summary functionality
                package = sys.argv[3] if len(sys.argv) > 3 else "lwt"
                print(f"Testing package summary for: {package}\n")
                result = await get_package_summary(package)
                print(json.dumps(result, indent=2))
            elif "--packages" in sys.argv:
                # Test unified search functionality
                query_idx = sys.argv.index("--test") + 1
                packages_idx = sys.argv.index("--packages") + 1
                query = sys.argv[query_idx] if query_idx < len(sys.argv) else "HTTP server"
                packages = sys.argv[packages_idx:] if packages_idx < len(sys.argv) else ["base", "lwt", "cohttp"]
                print(f"Testing unified search query: {query}")
                print(f"Packages: {packages}\n")
                result = await search_ocaml(query, packages)
                print(json.dumps(result, indent=2))
            elif "--sherlodoc" in sys.argv:
                # Test sherlodoc functionality
                sherlodoc_idx = sys.argv.index("--sherlodoc")
                query = sys.argv[sherlodoc_idx + 1] if sherlodoc_idx + 1 < len(sys.argv) else "Base.List.t"
                print(f"Testing sherlodoc query: {query}\n")
                result = await sherlodoc(query)
                print(json.dumps(result, indent=2))
            elif "--get-docs" in sys.argv:
                # Test get_module_docs functionality
                docs_idx = sys.argv.index("--get-docs")
                # New parameter order: module_path, package, library, version
                module_path = sys.argv[docs_idx + 1] if docs_idx + 1 < len(sys.argv) else "Gg-V3"
                package = sys.argv[docs_idx + 2] if docs_idx + 2 < len(sys.argv) else None
                library = sys.argv[docs_idx + 3] if docs_idx + 3 < len(sys.argv) else None
                version = sys.argv[docs_idx + 4] if docs_idx + 4 < len(sys.argv) else None
                
                if package and library:
                    if version:
                        print(f"Testing get_module_docs: {module_path} in {package}.{library} v{version}\n")
                    else:
                        print(f"Testing get_module_docs: {module_path} in {package}.{library} (latest version)\n")
                elif package:
                    print(f"Testing get_module_docs: {module_path} in package {package} (any library)\n")
                else:
                    print(f"Testing get_module_docs: {module_path} (searching all packages)\n")
                    
                result = await get_module_docs(module_path, package, library, version)
                if "content" in result:
                    # Truncate content for display
                    content = result["content"]
                    if len(content) > 500:
                        result["content"] = content[:500] + "... [truncated]"
                print(json.dumps(result, indent=2))
            else:
                # Test search functionality
                query = sys.argv[2] if len(sys.argv) > 2 else "HTTP server"
                print(f"Testing search query: {query}\n")
                result = await search_ocaml(query)
                print(json.dumps(result, indent=2))
        
        asyncio.run(test())
    else:
        # Run FastMCP server with HTTP SSE transport
        # Default SSE endpoint will be available at /sse
        # Using default port (probably 8000), may conflict with embedding server
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()