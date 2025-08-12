"""Markdown parsing utilities for OCaml documentation using mistune AST."""
import re
import mistune
from typing import Dict, List, Any, Optional, Tuple

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    return text.strip()

def parse_type_definition(name: str, signature: str) -> Dict[str, Any]:
    """Parse a type definition."""
    return {
        'kind': 'type',
        'name': name,
        'signature': signature,
        'anchor': f'type-{name}'
    }

def parse_value_definition(name: str, signature: str, doc: str = "") -> Dict[str, Any]:
    """Parse a value (function) definition."""
    return {
        'kind': 'value',
        'name': name,
        'signature': signature,
        'documentation': doc,
        'anchor': f'val-{name}'
    }

def extract_text_from_tokens(tokens: List[Dict[str, Any]]) -> str:
    """Extract text content from mistune tokens."""
    text_parts = []
    for token in tokens:
        if token.get('type') == 'text':
            text_parts.append(token.get('raw', ''))
        elif token.get('type') in ['code_inline', 'codespan']:
            text_parts.append('`' + token.get('raw', '') + '`')
        elif token.get('type') == 'strong' and 'children' in token:
            text_parts.append(extract_text_from_tokens(token['children']))
        elif token.get('type') == 'em' and 'children' in token:
            text_parts.append(extract_text_from_tokens(token['children']))
        elif token.get('type') == 'link' and 'children' in token:
            text_parts.append(extract_text_from_tokens(token['children']))
    return ''.join(text_parts).strip()

def parse_markdown_ast(content: str) -> List[Dict[str, Any]]:
    """Parse markdown content using mistune and extract elements."""
    # Create a markdown parser
    markdown_parser = mistune.create_markdown(renderer=None)
    
    # Parse to AST tokens - mistune returns a tuple, we want the first element
    parsed = markdown_parser.parse(content)
    tokens = parsed[0] if isinstance(parsed, tuple) else parsed
    
    elements = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        if token['type'] == 'heading':
            # Extract heading text
            title = extract_text_from_tokens(token.get('children', []))
            level = token.get('attrs', {}).get('level', 1)
            
            # Skip module title (h1 containing "Module")
            if level == 1 and 'Module' in title:
                i += 1
                continue
            
            section = {
                'kind': 'section',
                'title': title,
                'level': level,
                'content': ""
            }
            
            # Collect content until next heading or code block
            content_parts = []
            j = i + 1
            while j < len(tokens) and tokens[j]['type'] not in ['heading', 'block_code']:
                if tokens[j]['type'] == 'paragraph':
                    content_parts.append(extract_text_from_tokens(tokens[j].get('children', [])))
                j += 1
            
            section['content'] = '\n'.join(content_parts)
            elements.append(section)
            
        elif token.get('type') == 'block_code':
            code_text = token.get('raw', '').strip()
            
            if code_text and (code_text.startswith('val ') or 
                            code_text.startswith('type ') or 
                            code_text.startswith('module ')):
                # Get following documentation
                following_text = ""
                if i + 1 < len(tokens) and tokens[i + 1]['type'] == 'paragraph':
                    following_text = extract_text_from_tokens(tokens[i + 1].get('children', []))
                
                parsed = parse_ocaml_code_block(code_text, following_text)
                if parsed:
                    elements.append(parsed)
        
        i += 1
    
    return elements

def parse_ocaml_code_block(code_text: str, following_text: str = "") -> Optional[Dict[str, Any]]:
    """Parse a single OCaml code block."""
    code_text = code_text.strip()
    
    # Parse value definitions
    # Match both regular names (\w+) and special operator names in parentheses (\([^)]+\))
    val_match = re.match(r'val\s+(\w+|\([^)]+\))\s*:\s*(.+)', code_text, re.DOTALL)
    if val_match:
        name = val_match.group(1)
        sig = val_match.group(2).strip()
        return parse_value_definition(name, f"val {name} : {sig}", following_text)
    
    # Parse type definitions
    type_match = re.match(r'type\s+(\w+)', code_text)
    if type_match:
        name = type_match.group(1)
        return parse_type_definition(name, code_text)
    
    # Parse module type definitions
    module_type_match = re.match(r'module\s+type\s+(\w+)\s*=\s*sig\s+\.\.\.\s+end', code_text)
    if module_type_match:
        name = module_type_match.group(1)
        return {
            'name': name,
            'kind': 'module-type'
        }
    
    # Parse module definitions - handle various patterns
    # Pattern 1: module Name : sig ... end
    module_match1 = re.match(r'module\s+(\w+)\s*:\s*sig\s+\.\.\.\s+end', code_text)
    if module_match1:
        name = module_match1.group(1)
        return {
            'name': name,
            'kind': 'module'
        }
    
    # Pattern 2: module Name (Arg : Type) : ReturnType with constraints
    module_match2 = re.match(r'module\s+(\w+)\s*\([^)]+\)\s*:\s*\w+', code_text)
    if module_match2:
        name = module_match2.group(1)
        return {
            'name': name,
            'kind': 'module'
        }
    
    # Pattern 3: module Name : Type  (simple module signature)
    module_match3 = re.match(r'module\s+(\w+)\s*:\s*\w+', code_text)
    if module_match3:
        name = module_match3.group(1)
        return {
            'name': name,
            'kind': 'module'
        }
    
    return None

def parse_module_markdown(content: str) -> Dict[str, Any]:
    """Parse a markdown file containing OCaml module documentation."""
    result = {
        'elements': [],  # New ordered list of all elements
        'types': [],     # Keep for backward compatibility 
        'values': [],    # Keep for backward compatibility
        'modules': [],   # Keep for backward compatibility
        'module_documentation': "",
        'preamble': "",  # Documentation before first section or code element
        'sections': []   # Keep for backward compatibility
    }
    
    # Parse markdown to AST
    markdown_parser = mistune.create_markdown(renderer=None)
    parsed = markdown_parser.parse(content)
    tokens = parsed[0] if isinstance(parsed, tuple) else parsed
    
    # Extract module name and preamble
    module_name = None
    preamble_parts = []
    i = 0
    
    # Find the h1 with module name
    while i < len(tokens):
        token = tokens[i]
        if token.get('type') == 'heading' and token.get('attrs', {}).get('level') == 1:
            title = extract_text_from_tokens(token.get('children', []))
            match = re.search(r'Module `([^`]+)`', title)
            if match:
                module_name = match.group(1)
                i += 1
                break
        i += 1
    
    # Extract preamble (content after h1 until first h2+ or code block)
    while i < len(tokens):
        token = tokens[i]
        if token.get('type') == 'heading' and token.get('attrs', {}).get('level', 1) > 1:
            break
        elif token.get('type') == 'block_code':
            code_text = token.get('raw', '').strip()
            if code_text and (code_text.startswith('val ') or 
                            code_text.startswith('type ') or 
                            code_text.startswith('module ')):
                break
        elif token.get('type') == 'paragraph':
            preamble_parts.append(extract_text_from_tokens(token.get('children', [])))
        i += 1
    
    if preamble_parts:
        result['preamble'] = '\n'.join(preamble_parts).strip()
        result['module_documentation'] = result['preamble']  # backward compatibility
    
    # Process all elements using the AST
    elements = parse_markdown_ast(content)
    
    # Populate result with elements
    for elem in elements:
        result['elements'].append(elem)
        
        # Keep for backward compatibility
        if elem.get('kind') == 'value':
            result['values'].append(elem)
        elif elem.get('kind') == 'type':
            result['types'].append(elem)
        elif elem.get('kind') == 'module':
            result['modules'].append(elem)
        elif elem.get('kind') == 'section':
            result['sections'].append(elem)
    
    return result

def parse_markdown_documentation(file_path: str) -> Dict[str, Any]:
    """Parse a markdown documentation file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if this is a module documentation file
    if 'Module `' in content or '# Module' in content:
        return parse_module_markdown(content)
    
    # Otherwise, it's a general documentation file (README, CHANGES, etc.)
    return {
        'content': content,
        'type': 'documentation',
        'preamble': ''  # No preamble for non-module files
    }

def extract_module_path(file_path: str) -> List[str]:
    """Extract module path from file path."""
    # Example: docs-md/package/version/doc/module/Module-Submodule.md
    # Should return ['Module', 'Submodule']
    # Special case: Module-Submodule-module-type-TypeName.md
    # Should return ['Module', 'Submodule', 'TypeName']
    
    return extract_module_path_with_types(file_path)[0]

def extract_module_path_with_types(file_path: str) -> tuple[List[str], Dict[int, str]]:
    """Extract module path from file path and return type information.
    
    Returns:
        tuple: (module_path, type_info) where type_info maps indices to type kinds
               e.g. {1: 'module-type'} means position 1 in path is a module type
    """
    # Example: docs-md/package/version/doc/module/Module-Submodule.md
    # Should return (['Module', 'Submodule'], {})
    # Special case: Module-Submodule-module-type-TypeName-More.md
    # Should return (['Module', 'Submodule', 'TypeName', 'More'], {2: 'module-type'})
    
    parts = file_path.split('/')
    
    # Find the 'doc' directory
    try:
        doc_index = parts.index('doc')
    except ValueError:
        return ([], {})
    
    # Get the file name without extension
    if parts[-1].endswith('.md'):
        filename = parts[-1][:-3]  # Remove .md
    else:
        return ([], {})
    
    # If it's index.md, use the parent directory name
    if filename == 'index':
        if len(parts) > doc_index + 1:
            return ([parts[-2]], {})
        return ([], {})
    
    # Check for module-type pattern and handle specially
    if '-module-type-' in filename:
        # Split on '-module-type-' to separate the module path from the type name and rest
        module_part, rest = filename.split('-module-type-', 1)
        # Split the module part on '-' for nested modules
        module_parts = module_part.split('-') if module_part else []
        # The rest might contain the type name followed by more nested modules
        # Split the rest on '-' and add all parts
        rest_parts = rest.split('-') if rest else []
        
        # The first element in rest_parts is the module type
        type_info = {}
        if rest_parts:
            type_info[len(module_parts)] = 'module-type'
        
        module_parts.extend(rest_parts)
        return (module_parts, type_info)
    
    # Split on '-' for nested modules
    module_parts = filename.split('-')
    
    return (module_parts, {})