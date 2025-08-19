# OCaml MCP Server

This repository contains an OCaml MCP server, which allows coding agents like Claude to 
query the OCaml ecosystem and search for packages, libraries and modules that will help
them to complete coding tasks in OCaml. It is written in Python (for now!) and uses
the output of the [ocaml-docs-ci](https://github.com/ocurrent/ocaml-docs-ci) project
that builds the documentation hosted on [ocaml.org/p/](https://ocaml.org/p/)

## Running the server

In order to run the server, it's necessary to have a couple of other services running
first - sherlodoc and llama.cpp.

### Llama.cpp
llama.cpp server is for the embeddings. It should be run as follows:

```
$ llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF --embedding --pooling cls -ub 8192
```

by default this listens on port 8080 on localhost.

### sherlodoc

To run the sherlodoc server you'll first need access to the sherlodoc database.
See later in this doc for how to obtain this. It's not required unless you want this
aspect of the MCP server.

Clone `https://github.com/jonludlam/odoc` and checkout the `sherlodoc-www` branch.
Then:

```
$ cd sherlodoc/www
$ dune exec -- ./www.exe serve --db=/path/to/sherlodoc/db
```

## Regenerating the data

The data regeneration pipeline transforms OCaml documentation markdown files into searchable embeddings and indexes. This process involves several steps that must be run in sequence.

### Prerequisites

Before starting the regeneration process:

1. **Python Environment**: Set up the development environment using uv:
   ```bash
   uv sync
   ```

2. **Embedding Server**: For generating embeddings, run:
   ```bash
   llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF --embedding --pooling cls -ub 8192
   ```

### Step 0: Obtain the Markdown Documentation

The documentation extraction requires odoc-generated markdown files. Here's how to obtain them:

1. **Download Documentation Archives**
   ```bash
   # Example: Download documentation for a specific package
   wget http://sage.ci.dev/linked-live/p/lwt/5.9.1/content.tar
   
   # Extract the archive
   tar -xf content.tar
   # Creates: epoch-*/linked/p/lwt/5.9.1/
   ```

2. **Install Modified odoc**
   ```bash
   # Clone the specific branch with modified markdown support
   git clone https://github.com/jonludlam/odoc
   cd odoc
   git checkout odoc-llm-markdown
   
   # Build and install
   opam install . --deps-only
   dune build @install
   dune install
   ```

3. **Generate Markdown from odocl Files**
   ```bash
   # Navigate back to the odoc-llm directory
   cd ....../odoc-llm
   
   # Generate markdown documentation from the odocl files
   odoc markdown-generate -o ./docs-md-tmp

   # Move the docs into the right place
   mv ./docs-md-tmp/p ./docs-md
   ```

   Note: The odocl files embed version information that may need adjustment in odoc's source if version mismatches occur.

### Step 1: Parse Markdown Documentation

Convert the markdown documentation files into structured JSON format:

```bash
# Parse all packages
uv run extract_docs.py

# Parse specific packages
uv run python extract_docs.py --packages base lwt cohttp
```

This creates JSON files in `parsed-docs/` containing structured documentation data.

### Step 2: Generate Module Descriptions

Use an LLM via OpenRouter to generate semantic descriptions for each module. First generate a token and save it locally. In the
following examples, the token has been saved as the file 'token'

```bash
# Generate descriptions for all packages
uv run python generate_module_descriptions.py --llm-url https://openrouter.ai/api --model qwen/qwen3-235b-a22b --api-key-file token

# Generate for specific packages
uv run python generate_module_descriptions.py --package base --llm-url https://openrouter.ai/api --model qwen/qwen3-235b-a22b --api-key-file token

# Use a different model (e.g., Claude Sonnet for better quality)
uv run python generate_module_descriptions.py --llm-url https://openrouter.ai/api --model anthropic/claude-3.5-sonnet --workers 4 -api-key-file token

# Resume interrupted processing
uv run python generate_module_descriptions.py --resume --llm-url https://openrouter.ai/api --model qwen/qwen3-235b-a22b --api-key-file token
```

Popular model options on OpenRouter:
- `qwen/qwen3-235b-a22b` - Large model with excellent performance
- `anthropic/claude-3.5-sonnet` - Higher quality but more expensive
- `meta-llama/llama-3.1-70b-instruct` - Open source alternative

Output goes to `module-descriptions/` with one JSON file per package.

### Step 3: Generate Package Descriptions

Create concise package summaries from README content:

```bash
# Generate for all packages
uv run python generate_package_descriptions.py --llm-url https://openrouter.ai/api --model qwen/qwen3-235b-a22b --api-key-file token

# Generate for specific packages
uv run python generate_package_descriptions.py --package lwt --llm-url https://openrouter.ai/api --model qwen/qwen3-235b-a22b --api-key-file token

# With parallel processing (be mindful of rate limits)
uv run python generate_package_descriptions.py --llm-url https://openrouter.ai/api --model qwen/qwen3-235b-a22b --workers 4 --api-key-file token
```

Note: When using OpenRouter, reduce the number of workers to avoid rate limits. Most models allow 10-20 requests per minute.

Output goes to `package-descriptions/` directory.

### Step 4: Generate Module Embeddings

Create vector embeddings from module descriptions:

```bash
# Generate embeddings for all packages
uv run python generate_embeddings.py

# Process specific packages
uv run python generate_embeddings.py --packages base,core,lwt

# Resume from checkpoint
uv run python generate_embeddings.py --resume

# Custom configuration
uv run python generate_embeddings.py --workers 8 --batch-size 32
```

This creates compressed NPZ arrays in `package-embeddings/packages/{package_name}/`.

### Step 5: Generate Package Embeddings

Create embeddings for package-level descriptions:

```bash
# Generate package embeddings
uv run python generate_package_embeddings.py

# With custom settings
uv run python generate_package_embeddings.py --batch-size 16
```

Output goes to `package-embeddings/` directory.

### Step 6: Generate BM25 Indexes

Build full-text search indexes for keyword-based search:

```bash
# Build indexes for all packages
uv run python generate_module_index.py

# Build for specific packages
uv run python generate_module_index.py --packages base lwt cohttp

# Build first N packages for testing
uv run python generate_module_index.py --limit 10
```

Creates per-package BM25 indexes in `module-indexes/`.


### Monitoring Progress

- Each tool supports `--verbose` for detailed output
- Progress is saved in checkpoint files for resumable processing
- Check `checkpoint.json` files to see current progress
- Most tools skip already-processed packages by default

### Storage Requirements

- Parsed documentation: ~1.3GB
- Module descriptions: ~500MB
- Embeddings: ~600MB compressed
- BM25 indexes: ~400MB
- Total: ~3GB for complete dataset

### Troubleshooting

- **LLM timeouts**: Reduce `--workers` or increase timeout settings
- **Memory issues**: Process in smaller batches using `--limit`
- **Interrupted processing**: Most tools support `--resume` to continue
- **Missing packages**: Check `parsed-docs/` for parsing failures
