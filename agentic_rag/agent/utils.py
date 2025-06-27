import json
import re
import ast

def safe_json_parse(content: str, target_keys: list = None) -> dict:
    """
    Robust JSON/dict parser that handles both JSON and Python dict formats.
    
    Args:
        content: The text content to parse
        target_keys: List of keys that should be in the final result (for validation)
    
    Returns:
        Parsed dictionary or None if parsing fails
    """
    
    # Method 1: Try to find JSON blocks first
    json_patterns = [
        r"```json\s*(\{.*?\})\s*```",     # JSON code blocks
        r"```\s*(\{.*?\})\s*```",         # Generic code blocks  
        r"(\{(?:[^{}]|{[^{}]*})*\})",     # Better nested JSON matching
        r"(\{[^{}]*\"rating\"[^{}]*\})",  # Simple inline JSON with rating
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if target_keys is None or all(key in result for key in target_keys):
                    return result
            except json.JSONDecodeError:
                continue
    
    # Method 2: Try to find Python dict blocks and convert them
    dict_patterns = [
        r"(\{[^{}]*'[^']*'[^{}]*\})",  # Simple dicts with single quotes
        r"(\{.*?\})",                   # Any curly brace content
    ]
    
    for pattern in dict_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                # First try to convert single quotes to double quotes
                json_str = match.replace("'", '"')
                result = json.loads(json_str)
                if target_keys is None or all(key in result for key in target_keys):
                    return result
            except json.JSONDecodeError:
                try:
                    # Try using ast.literal_eval for Python dict syntax
                    result = ast.literal_eval(match)
                    if isinstance(result, dict):
                        if target_keys is None or all(key in result for key in target_keys):
                            return result
                except (ValueError, SyntaxError):
                    continue
    
    return None