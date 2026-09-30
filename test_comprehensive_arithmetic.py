#!/usr/bin/env python3
"""
Comprehensive arithmetic evaluation script.
Tests model capability across different difficulty levels and operations.
"""

import sys
import os
import numpy as np
sys.path.append('dataset')

from arithmetic_evaluation import decode_arithmetic_sequence, evaluate_arithmetic_equation


def analyze_dataset(dataset_path):
    """Analyze a dataset to understand its composition."""
    
    print(f"ANALYZING DATASET: {dataset_path}")
    print("="*60)
    
    # Load dataset
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Train directory not found: {train_dir}")
        return
        
    train_inputs = np.load(os.path.join(train_dir, 'all__inputs.npy'))
    train_labels = np.load(os.path.join(train_dir, 'all__labels.npy'))
    
    print(f"Training examples: {len(train_inputs)}")
    
    if os.path.exists(test_dir):
        test_inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'))
        test_labels = np.load(os.path.join(test_dir, 'all__labels.npy'))
        print(f"Test examples: {len(test_inputs)}")
    else:
        test_inputs, test_labels = None, None
        print("No test set found")
    
    # Analyze training data
    print("\\nTRAINING DATA ANALYSIS:")
    print("-" * 30)
    analyze_examples(train_inputs[:100], "Train (first 100)")
    
    # Analyze test data
    if test_inputs is not None:
        print("\\nTEST DATA ANALYSIS:")
        print("-" * 30)
        analyze_examples(test_inputs, "Test (all)")
    
    print("\\n" + "="*60)


def analyze_examples(inputs, dataset_name):
    """Analyze a set of examples."""
    
    stats = {
        'total': 0,
        'decoded': 0,
        'operations': {'+': 0, '-': 0, '*': 0, '/': 0},
        'operand_counts': {2: 0, 3: 0, 4: 0, 5: 0},
        'result_ranges': {
            'negative': 0,
            'zero': 0,
            'small_pos': 0,  # 1-100
            'medium_pos': 0, # 101-10000
            'large_pos': 0   # >10000
        },
        'difficulties': {
            'easy': 0,    # 1-2 digit operands
            'medium': 0,  # 3-4 digit operands  
            'hard': 0     # 5+ digit operands
        }
    }
    
    for i, input_seq in enumerate(inputs):
        stats['total'] += 1
        
        eq, result = decode_arithmetic_sequence(input_seq)
        if not eq:
            continue
            
        stats['decoded'] += 1
        
        # Count operations
        for op in ['+', '-', '*', '/']:
            if f' {op} ' in eq:
                stats['operations'][op] += 1
                break
        
        # Count operands
        operand_count = eq.count(' ') // 2 + 1  # Rough estimate
        if operand_count in stats['operand_counts']:
            stats['operand_counts'][operand_count] += 1
        
        # Categorize results
        if result is not None:
            if result < 0:
                stats['result_ranges']['negative'] += 1
            elif result == 0:
                stats['result_ranges']['zero'] += 1
            elif 1 <= result <= 100:
                stats['result_ranges']['small_pos'] += 1
            elif 101 <= result <= 10000:
                stats['result_ranges']['medium_pos'] += 1
            else:
                stats['result_ranges']['large_pos'] += 1
        
        # Estimate difficulty by largest operand
        parts = eq.replace('=', '').split()
        numbers = [abs(int(p)) for p in parts if p.isdigit() or (p.startswith('-') and p[1:].isdigit())]
        if numbers:
            max_operand = max(numbers)
            if max_operand < 100:
                stats['difficulties']['easy'] += 1
            elif max_operand < 10000:
                stats['difficulties']['medium'] += 1
            else:
                stats['difficulties']['hard'] += 1
    
    # Print analysis
    print(f"{dataset_name}: {stats['decoded']}/{stats['total']} decoded successfully")
    
    if stats['decoded'] > 0:
        print("\\nOperations:")
        for op, count in stats['operations'].items():
            pct = count / stats['decoded'] * 100
            print(f"  {op}: {count} ({pct:.1f}%)")
        
        print("\\nOperand counts:")
        for count, freq in stats['operand_counts'].items():
            if freq > 0:
                pct = freq / stats['decoded'] * 100
                print(f"  {count} operands: {freq} ({pct:.1f}%)")
        
        print("\\nResult ranges:")
        for range_name, count in stats['result_ranges'].items():
            pct = count / stats['decoded'] * 100
            print(f"  {range_name}: {count} ({pct:.1f}%)")
        
        print("\\nDifficulties:")
        for diff, count in stats['difficulties'].items():
            pct = count / stats['decoded'] * 100
            print(f"  {diff}: {count} ({pct:.1f}%)")
    
    # Show sample problems
    print("\\nSample problems:")
    samples_shown = 0
    for input_seq in inputs:
        if samples_shown >= 5:
            break
        eq, result = decode_arithmetic_sequence(input_seq)
        if eq:
            is_correct, expected = evaluate_arithmetic_equation(eq)
            print(f"  {eq} -> Expected: {expected}, Correct: {is_correct}")
            samples_shown += 1


def compare_datasets(dataset_paths):
    """Compare multiple datasets."""
    
    print("DATASET COMPARISON")
    print("="*60)
    
    for path in dataset_paths:
        if os.path.exists(path):
            analyze_dataset(path)
            print()
        else:
            print(f"Dataset not found: {path}")
            print()


def create_manual_test_suite():
    """Create a manual test suite with known correct answers."""
    
    test_cases = [
        # Basic operations - Easy
        ("7 + 8 = 15", True, 15),
        ("23 + 45 = 68", True, 68),
        ("50 - 23 = 27", True, 27),
        ("6 * 7 = 42", True, 42),
        ("84 / 12 = 7", True, 7),
        
        # Wrong answers
        ("7 + 8 = 16", False, 15),  # Wrong result
        ("10 * 5 = 51", False, 50), # Wrong result
        
        # Multi-operand
        ("10 + 20 + 30 = 60", True, 60),
        ("1000 - 200 - 300 = 500", True, 500),
        ("2 * 3 * 4 = 24", True, 24),
        
        # Negative numbers
        ("-50 + 30 = -20", True, -20),
        ("100 - -25 = 125", True, 125),
        ("-8 * 6 = -48", True, -48),
        
        # Edge cases
        ("0 + 5 = 5", True, 5),
        ("10 * 0 = 0", True, 0),
        ("1 * 1 = 1", True, 1),
    ]
    
    return test_cases


def test_evaluation_function():
    """Test the evaluation function with known cases."""
    
    print("TESTING EVALUATION FUNCTION")
    print("="*60)
    
    test_cases = create_manual_test_suite()
    
    correct = 0
    total = len(test_cases)
    
    for equation, expected_correctness, expected_result in test_cases:
        is_correct, actual_result = evaluate_arithmetic_equation(equation)
        
        # Check if our evaluation matches expectations
        eval_correct = (is_correct == expected_correctness and 
                       actual_result == expected_result)
        
        if eval_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {equation}")
        print(f"    Expected: correct={expected_correctness}, result={expected_result}")
        print(f"    Got: correct={is_correct}, result={actual_result}")
        print()
    
    print(f"Evaluation function accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*60)


def main():
    """Main function."""
    
    # Test evaluation function
    test_evaluation_function()
    print("\\n")
    
    # Compare different datasets
    datasets_to_compare = [
        "data/arithmetic-6digit-200-aug-200-final",  # Original problematic dataset
        "data/arithmetic-enhanced-balanced",          # New enhanced dataset
    ]
    
    compare_datasets(datasets_to_compare)


if __name__ == "__main__":
    main()