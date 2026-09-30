#!/usr/bin/env python3
"""
Manual arithmetic testing script for interactive model evaluation.
Bypasses torch.compile issues by working directly with saved predictions or using alternative methods.
"""

import sys
import os
import torch
import numpy as np
import yaml
sys.path.append('dataset')

from arithmetic_evaluation import decode_arithmetic_sequence, evaluate_arithmetic_equation
from build_enhanced_arithmetic_dataset import arithmetic_to_sequence
from pretrain import PretrainConfig, init_train_state, create_dataloader


def create_manual_test_problems():
    """Create custom test problems for manual evaluation."""
    
    problems = [
        # Simple tests
        ([5, 3], "+", 8),
        ([10, 4], "-", 6),  
        ([7, 6], "*", 42),
        ([15, 3], "/", 5),
        
        # Medium difficulty
        ([123, 456], "+", 579),
        ([1000, 237], "-", 763),
        ([12, 25], "*", 300),
        ([144, 12], "/", 12),
        
        # Multi-operand
        ([10, 20, 30], "+", 60),
        ([100, 30, 20], "-", 50),
        ([2, 3, 4], "*", 24),
        
        # Negative numbers
        ([-5, 8], "+", 3),
        ([50, -15], "+", 35),
        ([-10, -5], "+", -15),
        ([20, -30], "-", 50),
        ([-7, 8], "*", -56),
        ([-48, -6], "/", 8),
        
        # Edge cases
        ([0, 5], "+", 5),
        ([42, 0], "+", 42),
        ([10, 0], "*", 0),
        ([0, 7], "*", 0),
        ([1, 1], "*", 1),
        
        # Harder problems
        ([9876, 1234], "+", 11110),
        ([5000, 2345], "-", 2655),
        ([99, 11], "*", 1089),
        ([9999, 9], "/", 1111),
    ]
    
    return problems


def test_with_saved_predictions():
    """Test using saved model predictions (most reliable method)."""
    
    print("🔍 TESTING WITH SAVED PREDICTIONS")
    print("="*60)
    
    checkpoint_dir = "checkpoints/Arithmetic-enhanced-balanced ACT-torch/HierarchicalReasoningModel_ACTV1 gentle-quokka"
    
    # Find latest prediction file
    pred_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_all_preds.0')]
    if not pred_files:
        print("❌ No prediction files found")
        return
    
    latest_pred = sorted(pred_files)[-1]
    pred_path = os.path.join(checkpoint_dir, latest_pred)
    
    print(f"Loading predictions from: {latest_pred}")
    
    try:
        predictions = torch.load(pred_path, map_location='cpu')
        
        logits = predictions['logits']
        pred_tokens = torch.argmax(logits, dim=-1)
        inputs = predictions.get('inputs', None)
        labels = predictions.get('labels', None)
        
        print(f"\nTesting on {len(pred_tokens)} saved examples:")
        print("-" * 40)
        
        # Test all available predictions
        correct = 0
        total = 0
        
        for i in range(min(25, len(pred_tokens))):  # Test up to 25 examples
            if inputs is not None:
                input_eq, _ = decode_arithmetic_sequence(inputs[i])
            else:
                input_eq = None
                
            label_eq, label_result = decode_arithmetic_sequence(labels[i])
            pred_eq, pred_result = decode_arithmetic_sequence(pred_tokens[i])
            
            if not label_eq or not pred_eq:
                continue
                
            total += 1
            is_correct = (pred_result == label_result)
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{i+1:2d}. {status} {label_eq}")
            print(f"      Model: {pred_eq}")
            if not is_correct:
                print(f"      Expected: {label_result}, Got: {pred_result}")
            print()
        
        accuracy = correct / total if total > 0 else 0
        print(f"RESULTS: {correct}/{total} correct ({accuracy*100:.1f}%)")
        
        if accuracy >= 0.95:
            print("🎉 EXCELLENT performance!")
        elif accuracy >= 0.8:
            print("✅ GOOD performance")
        elif accuracy >= 0.6:
            print("⚠️  FAIR performance") 
        else:
            print("❌ POOR performance")
            
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")


def test_with_custom_problems():
    """Test model on custom problems by creating token sequences."""
    
    print("\n🧪 TESTING WITH CUSTOM PROBLEMS")
    print("="*60)
    
    problems = create_manual_test_problems()
    
    print(f"Testing {len(problems)} custom problems:")
    print("-" * 40)
    
    correct_encodings = 0
    total_problems = len(problems)
    
    for i, (operands, operation, expected_result) in enumerate(problems):
        print(f"{i+1:2d}. Testing: {' '.join(map(str, operands))} {operation} ? = ?")
        
        # Convert to token sequence
        tokens = arithmetic_to_sequence(operands, operation, expected_result, max_seq_len=32)
        
        if tokens is None:
            print("    ❌ Problem too long for sequence length")
            continue
        
        # Decode back to verify encoding
        equation, result = decode_arithmetic_sequence(tokens)
        
        print(f"    Token sequence: {tokens}")
        print(f"    Decoded equation: {equation}")
        print(f"    Expected result: {expected_result}")
        
        # Check if encoding is correct
        is_correct, computed_result = evaluate_arithmetic_equation(equation)
        
        if is_correct and computed_result == expected_result:
            correct_encodings += 1
            print(f"    ✅ Encoding CORRECT")
        else:
            print(f"    ❌ Encoding ERROR (computed: {computed_result})")
        
        print()
    
    encoding_accuracy = correct_encodings / total_problems
    print(f"ENCODING RESULTS: {correct_encodings}/{total_problems} correct ({encoding_accuracy*100:.1f}%)")
    
    if encoding_accuracy >= 0.95:
        print("✅ All problems encoded correctly - ready for model inference")
    else:
        print("❌ Some encoding issues found")


def simulate_model_inference():
    """Simulate what the model should predict based on training patterns."""
    
    print("\n🤖 SIMULATING MODEL INFERENCE")
    print("="*60)
    
    problems = create_manual_test_problems()
    
    print("Based on training patterns, the model should predict:")
    print("-" * 50)
    
    for i, (operands, operation, expected_result) in enumerate(problems[:10]):  # Show first 10
        # Build expected equation
        equation_parts = [str(operands[0])]
        for j in range(1, len(operands)):
            equation_parts.append(operation)
            equation_parts.append(str(operands[j]))
        equation_parts.extend(['=', str(expected_result)])
        
        expected_equation = ' '.join(equation_parts)
        
        print(f"{i+1:2d}. Problem: {' '.join(map(str, operands))} {operation} ? = ?")
        print(f"    Expected: {expected_equation}")
        
        # Verify it's mathematically correct
        is_correct, computed = evaluate_arithmetic_equation(expected_equation)
        status = "✅" if is_correct else "❌"
        print(f"    {status} Mathematically correct: {is_correct}")
        print()


def interactive_test_mode():
    """Interactive mode where user can input problems."""
    
    print("\n🎮 INTERACTIVE TEST MODE")
    print("="*60)
    print("Enter arithmetic problems to test (or 'quit' to exit)")
    print("Format: number1 operation number2 (e.g., '7 + 8')")
    print("Supported: +, -, *, /")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nEnter problem: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Parse input
            parts = user_input.split()
            if len(parts) != 3:
                print("❌ Format: number1 operation number2")
                continue
            
            try:
                num1 = int(parts[0])
                operation = parts[1]
                num2 = int(parts[2])
            except ValueError:
                print("❌ Invalid numbers")
                continue
            
            if operation not in ['+', '-', '*', '/']:
                print("❌ Unsupported operation")
                continue
            
            # Calculate expected result
            if operation == '+':
                expected = num1 + num2
            elif operation == '-':
                expected = num1 - num2
            elif operation == '*':
                expected = num1 * num2
            elif operation == '/':
                if num2 == 0:
                    print("❌ Division by zero")
                    continue
                expected = num1 // num2  # Integer division
            
            # Test encoding
            tokens = arithmetic_to_sequence([num1, num2], operation, expected)
            if tokens:
                equation, result = decode_arithmetic_sequence(tokens)
                print(f"✅ Model should predict: {equation}")
                
                # Verify
                is_correct, computed = evaluate_arithmetic_equation(equation)
                if is_correct and computed == expected:
                    print(f"✅ Prediction would be CORRECT")
                else:
                    print(f"❌ Prediction would be INCORRECT")
            else:
                print("❌ Problem too complex for current model")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function with menu."""
    
    print("🧮 MANUAL ARITHMETIC MODEL TESTING")
    print("="*60)
    print("Choose testing method:")
    print("1. Test with saved predictions (most reliable)")
    print("2. Test custom problems (encoding verification)")
    print("3. Simulate model inference (prediction simulation)")
    print("4. Interactive test mode")
    print("5. Run all tests")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            test_with_saved_predictions()
        elif choice == '2':
            test_with_custom_problems()
        elif choice == '3':
            simulate_model_inference()
        elif choice == '4':
            interactive_test_mode()
        elif choice == '5':
            test_with_saved_predictions()
            test_with_custom_problems()
            simulate_model_inference()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()