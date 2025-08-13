#!/usr/bin/env python3
"""
Test script for arithmetic evaluation functionality.

This script tests the arithmetic evaluation utilities to ensure correctness
checking works properly during training.
"""

import sys
import os

# Add dataset path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset'))

from dataset.arithmetic_evaluation import (
    decode_arithmetic_sequence,
    evaluate_arithmetic_equation
)


def test_token_decoding():
    """Test token sequence decoding."""
    print("=== Testing Token Decoding ===")
    
    # Test case: -1860 * 6191 = -11515260
    # Tokens: NEG 1 8 6 0 * 6 1 9 1 = NEG 1 1 5 1 5 2 6 0 EOS
    tokens = [17, 2, 9, 7, 1, 13, 7, 2, 10, 2, 15, 17, 2, 2, 6, 2, 6, 3, 7, 1, 16]
    
    equation, result = decode_arithmetic_sequence(tokens)
    print(f"Decoded equation: {equation}")
    print(f"Decoded result: {result}")
    
    # Test evaluation
    is_correct, expected = evaluate_arithmetic_equation(equation)
    print(f"Is correct: {is_correct}, Expected: {expected}")
    
    # Verify manually
    manual_result = -1860 * 6191
    print(f"Manual calculation: {manual_result}")
    print(f"Match: {result == manual_result}")


def test_simple_examples():
    """Test simple arithmetic examples."""
    print("\n=== Testing Simple Examples ===")
    
    test_cases = [
        # Simple addition: 12 + 34 = 46
        ([2, 3, 11, 4, 5, 15, 5, 7, 16], "12 + 34 = 46"),
        
        # Simple subtraction: 100 - 25 = 75
        ([2, 1, 1, 12, 3, 6, 15, 8, 6, 16], "100 - 25 = 75"),
        
        # Simple multiplication: 7 * 8 = 56
        ([8, 13, 9, 15, 6, 7, 16], "7 * 8 = 56"),
        
        # Division: 24 / 6 = 4
        ([3, 5, 14, 7, 15, 5, 16], "24 / 6 = 4"),
        
        # Negative result: 5 - 10 = -5
        ([6, 12, 2, 1, 15, 17, 6, 16], "5 - 10 = -5"),
    ]
    
    for i, (tokens, expected_desc) in enumerate(test_cases):
        print(f"\nTest case {i+1}: {expected_desc}")
        equation, result = decode_arithmetic_sequence(tokens)
        print(f"  Decoded: {equation}")
        
        if equation:
            is_correct, expected_result = evaluate_arithmetic_equation(equation)
            print(f"  Correct: {is_correct}, Expected: {expected_result}, Got: {result}")
        else:
            print(f"  Failed to decode tokens: {tokens}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    # Test empty/invalid sequences
    empty_tokens = [0, 0, 0, 0, 0]  # All PAD
    eq, result = decode_arithmetic_sequence(empty_tokens)
    print(f"Empty sequence: equation='{eq}', result={result}")
    
    # Test malformed sequence
    malformed = [11, 15, 12, 16]  # + = - EOS (nonsense)
    eq, result = decode_arithmetic_sequence(malformed)
    print(f"Malformed sequence: equation='{eq}', result={result}")
    
    # Test evaluation of malformed equation
    if eq:
        is_correct, expected = evaluate_arithmetic_equation(eq)
        print(f"Malformed evaluation: correct={is_correct}, expected={expected}")
    
    # Test wrong answer
    wrong_answer = [2, 3, 11, 4, 5, 15, 9, 9, 16]  # 12 + 34 = 99 (wrong!)
    eq, result = decode_arithmetic_sequence(wrong_answer)
    print(f"Wrong answer: {eq}")
    if eq:
        is_correct, expected = evaluate_arithmetic_equation(eq)
        print(f"  Correct: {is_correct}, Expected: {expected}, Got: {result}")


def main():
    """Run all tests."""
    print("Testing Arithmetic Evaluation System")
    print("=" * 50)
    
    test_token_decoding()
    test_simple_examples()
    test_edge_cases()
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    main()