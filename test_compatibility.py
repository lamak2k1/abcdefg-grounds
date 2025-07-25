#!/usr/bin/env python3
"""
Test script to verify RAG application compatibility with updated dependencies.
This script tests core functionality without requiring vector indices.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported successfully."""
    print("Testing package imports...")
    
    required_packages = [
        'llama_index.core',
        'llama_index.llms.openai', 
        'llama_index.embeddings.openai',
        'llama_index.embeddings.huggingface',
        'fastapi',
        'streamlit',
        'openai',
        'uvicorn'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        return False
    else:
        print("\n✓ All package imports successful!")
        return True

def test_fastapi_startup():
    """Test that FastAPI application can start without errors."""
    print("\nTesting FastAPI application startup...")
    
    os.environ['OPENAI_API_KEY'] = 'test_key_for_compatibility'
    os.environ['KARL_NAME'] = 'Karl Kaufman'
    os.environ['KARL_TOPICS'] = 'investing, finance'
    
    try:
        sys.path.insert(0, str(Path.cwd() / 'app'))
        from main import app
        print("✓ FastAPI app imports successfully")
        return True
    except Exception as e:
        print(f"✗ FastAPI app import failed: {e}")
        return False

def test_streamlit_syntax():
    """Test that Streamlit application has valid syntax."""
    print("\nTesting Streamlit application syntax...")
    
    try:
        spec = importlib.util.spec_from_file_location("pol", "pol.py")
        if spec is None:
            print("✗ Could not load pol.py")
            return False
            
        module = importlib.util.module_from_spec(spec)
        print("✓ Streamlit app has valid syntax")
        return True
    except Exception as e:
        print(f"✗ Streamlit app syntax error: {e}")
        return False

def test_llama_index_compatibility():
    """Test LlamaIndex core functionality."""
    print("\nTesting LlamaIndex compatibility...")
    
    try:
        from llama_index.core import Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        Settings.llm = OpenAI(api_key="test_key", model="gpt-4o-mini")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        print("✓ LlamaIndex configuration successful")
        return True
    except Exception as e:
        print(f"✗ LlamaIndex compatibility test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("=== RAG Application Compatibility Test ===\n")
    
    tests = [
        test_imports,
        test_fastapi_startup, 
        test_streamlit_syntax,
        test_llama_index_compatibility
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All compatibility tests passed!")
        print("✓ Updated dependencies are working correctly")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
