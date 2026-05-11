#!/usr/bin/env python
"""
Test script for the Fracture Detection API
"""
import requests
import sys
from pathlib import Path

API_BASE = 'http://localhost:8000'


def test_health():
    """Test health check endpoint"""
    print('Testing /health endpoint...')
    try:
        response = requests.get(f'{API_BASE}/health', timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f'✓ Health check passed: {data}')
        return True
    except Exception as e:
        print(f'✗ Health check failed: {e}')
        return False


def test_predict(image_path):
    """Test prediction endpoint with an image"""
    print(f'\nTesting /predict endpoint with {image_path}...')
    
    if not Path(image_path).exists():
        print(f'✗ Image file not found: {image_path}')
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{API_BASE}/predict', files=files, timeout=30)
            response.raise_for_status()
            data = response.json()
            print(f'✓ Prediction result:')
            print(f'  Prediction: {data["prediction"]}')
            print(f'  Confidence: {data["confidence"]}%')
            return True
    except Exception as e:
        print(f'✗ Prediction failed: {e}')
        return False


if __name__ == '__main__':
    print('=' * 50)
    print('Fracture Detection API - Test Suite')
    print('=' * 50)
    
    # Test health
    if not test_health():
        print('\n✗ Backend server is not running!')
        print(f'Start the backend with:')
        print(f'  uvicorn backend.main:app --reload')
        sys.exit(1)
    
    # Test prediction if image provided
    if len(sys.argv) > 1:
        test_predict(sys.argv[1])
    else:
        print('\nTo test prediction, provide an image path:')
        print('  python backend/test_api.py path/to/image.jpg')
    
    print('\n' + '=' * 50)
    print('Tests completed!')
