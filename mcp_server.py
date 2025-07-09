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
from semantic_search import SemanticSearch
from unified_search import UnifiedSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ocaml-search", host="0.0.0.0")

# Global search engine instances (lazy loaded)
search_engine: Optional[SemanticSearch] = None
unified_engine: Optional[UnifiedSearchEngine] = None
embeddings_dir = Path("package-embeddings")
package_descriptions_dir = Path("package-descriptions")
indexes_dir = Path("module-indexes")
module_descriptions_dir = Path("module-descriptions")

@mcp.tool()
async def find_ocaml_packages(functionality: str) -> Dict[str, Any]:
    """
    Discover OCaml packages across the entire ecosystem for specific functionality.
    
    Use this when you don't know which packages might contain what you need.
    This searches across all available packages to find the most relevant ones.
    
    Args:
        functionality: Describe what you're looking for. Be specific:
                      - "WebSocket client implementation"
                      - "Machine learning matrix operations" 
                      - "CSV file parsing and writing"
                      - "OAuth2 authentication flow"
                      - "Image processing and filtering"
    
    Returns:
        List of packages ranked by relevance, each with:
        - package: Package name
        - module: Primary module providing the functionality  
        - description: What the module does
    """
    global search_engine
    
    # Initialize search engine on first use (lazy loading)
    if search_engine is None:
        logger.info("Initializing semantic search engine...")
        try:
            search_engine = SemanticSearch(embeddings_dir)
        except Exception as e:
            error_msg = f"Failed to initialize search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Perform semantic search
        results = search_engine.search(functionality, top_k=5)
        
        # Format results for MCP response
        return {
            "query": functionality,
            "packages": [
                {
                    "package": result["package"],
                    "library": result.get("library"),
                    "module": result["module_path"],
                    "description": result["description"]
                }
                for result in results
            ]
        }
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


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
async def search_ocaml_modules(query: str, packages: List[str], top_k: int = 8) -> Dict[str, Any]:
    """
    Find OCaml modules that provide specific functionality within chosen packages.
    
    Provide a clear description of the functionality you need and specify which 
    packages to search. The tool will find relevant modules using both conceptual 
    understanding and exact keyword matching.
    
    Args:
        query: Specific functionality you're looking for. Be precise about what you need:
               - "MD5 hash function" 
               - "HTTP client for making requests"
               - "JSON parsing and serialization"
               - "list sorting operations"
               - "TCP socket server"
        packages: List of OCaml package names to search within. You must specify
                 which packages to search - common ones include:
                 ['base', 'core', 'lwt', 'async', 'cohttp', 'yojson', 'cmdliner']
        top_k: Maximum number of results to return (default: 8)
    
    Returns:
        Two lists of matching modules:
        - semantic_results: Modules with conceptually similar functionality (includes descriptions)
        - keyword_results: Modules containing your exact keywords in their documentation
        
        Each result includes the package name, module name, and module path.
    """
    global unified_engine
    
    if not packages:
        return {"error": "No packages specified. Please provide a list of package names to search."}
    
    # Initialize unified search engine on first use (lazy loading)
    if unified_engine is None:
        logger.info("Initializing unified search engine...")
        try:
            unified_engine = UnifiedSearchEngine(
                embedding_dir=str(embeddings_dir),
                index_dir=str(indexes_dir)
            )
        except Exception as e:
            error_msg = f"Failed to initialize unified search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Perform unified search - split top_k between the two methods
        results_per_method = top_k // 2
        results = unified_engine.unified_search(query, packages, results_per_method)
        
        # Format results for MCP response
        return {
            "query": results["query"],
            "packages_searched": results["packages_searched"],
            "semantic_results": [
                {
                    "package": r["package"],
                    "library": r.get("library"),
                    "module": r["module_name"],
                    "module_path": r["module_path"],
                    "description": r["description"]
                }
                for r in results["embedding_results"]
            ],
            "keyword_results": [
                {
                    "package": r["package"],
                    "library": r.get("library"),
                    "module": r["module_name"],
                    "module_path": r["module_path"]
                }
                for r in results["bm25_results"]
            ],
            "total_results": len(results["embedding_results"]) + len(results["bm25_results"])
        }
        
    except Exception as e:
        error_msg = f"Unified search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


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
                result = await search_ocaml_modules(query, packages)
                print(json.dumps(result, indent=2))
            elif "--sherlodoc" in sys.argv:
                # Test sherlodoc functionality
                sherlodoc_idx = sys.argv.index("--sherlodoc")
                query = sys.argv[sherlodoc_idx + 1] if sherlodoc_idx + 1 < len(sys.argv) else "Base.List.t"
                print(f"Testing sherlodoc query: {query}\n")
                result = await sherlodoc(query)
                print(json.dumps(result, indent=2))
            else:
                # Test search functionality
                query = sys.argv[2] if len(sys.argv) > 2 else "HTTP server"
                print(f"Testing search query: {query}\n")
                result = await find_ocaml_packages(query)
                print(json.dumps(result, indent=2))
        
        asyncio.run(test())
    else:
        # Run FastMCP server with HTTP SSE transport
        # Default SSE endpoint will be available at /sse
        # Using default port (probably 8000), may conflict with embedding server
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()