#!/usr/bin/env python3
"""
Test script to call the Python sandbox API and execute matrix multiplication.
"""

import json
import sys
import time

import requests

# Default server URL
DEFAULT_URL = "http://localhost:8080"


def test_health(url: str):
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed!")
        print(f"Status: {data.get('status')}")
        print(f"Python version: {data.get('python_version', 'N/A')}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_matrix_multiplication(url: str):
    """Test matrix multiplication via API"""
    print("\n" + "=" * 60)
    print("Testing Matrix Multiplication via API")
    print("=" * 60)

    # Matrix multiplication code with return value
    code = """
import numpy as np
import time

print("=" * 60)
print("Matrix Multiplication Test")
print("=" * 60)

# Test 1: Small matrices (100x100)
print("\\nTest 1: Small matrices (100x100)")
size1 = 100
A = np.random.rand(size1, size1)
B = np.random.rand(size1, size1)

start_time = time.time()
C = np.dot(A, B)
elapsed_time = time.time() - start_time

print(f"Matrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")
print(f"Result C shape: {C.shape}")
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"First element of result: {C[0, 0]:.6f}")

# Test 2: Medium matrices (500x500)
print("\\nTest 2: Medium matrices (500x500)")
size2 = 500
A = np.random.rand(size2, size2)
B = np.random.rand(size2, size2)

start_time = time.time()
C = np.dot(A, B)
elapsed_time = time.time() - start_time

print(f"Matrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")
print(f"Result C shape: {C.shape}")
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"First element of result: {C[0, 0]:.6f}")

# Test 3: Verify correctness with known matrices
print("\\nTest 3: Verification with known matrices")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
expected = np.array([[19, 22], [43, 50]])
C = np.dot(A, B)

print(f"Matrix A:\\n{A}")
print(f"Matrix B:\\n{B}")
print(f"Result C:\\n{C}")
print(f"Expected:\\n{expected}")

verification_passed = np.allclose(C, expected)
if verification_passed:
    print("✅ Verification passed!")
else:
    print("❌ Verification failed!")

# Test 4: Using @ operator (Python 3.5+)
print("\\nTest 4: Using @ operator")
A = np.random.rand(200, 200)
B = np.random.rand(200, 200)

start_time = time.time()
C = A @ B
elapsed_time = time.time() - start_time

print(f"Matrix shape: {A.shape}")
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"First element of result: {C[0, 0]:.6f}")

print("\\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
"""

    payload = {"code": code, "timeout": 300}

    try:
        print("\nSending request to server...")
        start_time = time.time()
        response = requests.post(
            f"{url}/execute",
            json=payload,
            timeout=320,  # Slightly longer than code timeout
        )
        elapsed_time = time.time() - start_time

        response.raise_for_status()
        result = response.json()

        print(f"\n✅ Request completed in {elapsed_time:.2f} seconds")
        print(f"Success: {result.get('success')}")
        print(f"Exit code: {result.get('exit_code')}")

        if result.get("success"):
            print("\n" + "-" * 60)
            print("Output:")
            print("-" * 60)
            print(result.get("output", ""))
            print("-" * 60)

            # Display return value if present
            return_value = result.get("return_value")
            if return_value is not None:
                print("\n" + "-" * 60)
                print("Return Value:")
                print("-" * 60)
                if isinstance(return_value, dict):
                    print(json.dumps(return_value, indent=2))
                else:
                    print(return_value)
                print("-" * 60)
        else:
            print("\n" + "-" * 60)
            print("Error:")
            print("-" * 60)
            print(result.get("error", ""))
            print("-" * 60)
            return False

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False


def main():
    """Main function"""
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    print(f"Testing Python Sandbox API at: {url}")
    print()

    # Test health
    if not test_health(url):
        print("\n❌ Health check failed. Is the server running?")
        print("Start the server with: ./start_background.sh")
        sys.exit(1)

    # Test matrix multiplication
    if test_matrix_multiplication(url):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Matrix multiplication test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
