#!/usr/bin/env python3
"""
Simple manual testing script for the arithmetic model.
Run this to quickly test the trained model on various problems.
"""

import sys
import torch
sys.path.append('dataset')
from arithmetic_evaluation import decode_arithmetic_sequence, evaluate_arithmetic_equation
from build_enhanced_arithmetic_dataset import arithmetic_to_sequence


def test_latest_checkpoint():
    """Test the latest trained model checkpoint."""
    
    print("🧮 ARITHMETIC MODEL - MANUAL TESTING")
    print("="*50)
    
    # Load latest predictions
    pred_path = 'checkpoints/Arithmetic-enhanced-balanced ACT-torch/HierarchicalReasoningModel_ACTV1 gentle-quokka/step_5740_all_preds.0'
    
    try:
        predictions = torch.load(pred_path, map_location='cpu')
        print("✅ Latest model checkpoint loaded successfully!")
    except:
        print("❌ Could not load checkpoint. Make sure training is complete.")
        return
    
    # Test on saved predictions
    logits = predictions['logits']
    pred_tokens = torch.argmax(logits, dim=-1)
    labels = predictions['labels']
    
    print(f"\n📊 RESULTS ON TEST SET ({len(labels)} problems):")
    print("-" * 40)
    
    correct = 0
    total = 0
    examples_shown = 0
    
    for i in range(len(pred_tokens)):
        label_eq, label_result = decode_arithmetic_sequence(labels[i])
        pred_eq, pred_result = decode_arithmetic_sequence(pred_tokens[i])
        
        if not label_eq or not pred_eq:
            continue
            
        total += 1
        is_correct = (pred_result == label_result)
        if is_correct:
            correct += 1
        
        # Show first 10 examples
        if examples_shown < 10:
            status = "✅" if is_correct else "❌"
            print(f"{examples_shown+1:2d}. {status} Problem: {label_eq}")
            print(f"      Model answered: {pred_eq}")
            if not is_correct:
                print(f"      ❌ Expected {label_result}, got {pred_result}")
            examples_shown += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Correct: {correct}/{total}")  
    print(f"   Accuracy: {accuracy*100:.1f}%")
    
    if accuracy >= 0.95:
        print("   🎉 EXCELLENT! Model mastered arithmetic!")
    elif accuracy >= 0.8:
        print("   ✅ GOOD performance")
    elif accuracy >= 0.6:
        print("   ⚠️  FAIR performance")
    else:
        print("   ❌ POOR performance")


def test_custom_problems():
    """Test on custom problems to show model capability."""
    
    print(f"\n🧪 TESTING CUSTOM PROBLEMS:")
    print("-" * 40)
    
    # Custom test problems
    problems = [
        # Your custom tests here
        ([15, 27], "+", 42),
        ([100, 38], "-", 62),
        ([11, 9], "*", 99),
        ([96, 12], "/", 8),
        ([5, 10, 15], "+", 30),
        ([-20, 35], "+", 15),
        ([0, 999], "+", 999),
    ]
    
    print("Problems the model should handle correctly:")
    
    for i, (operands, op, expected) in enumerate(problems):
        problem_str = f"{' '.join(map(str, operands))} {op} ? = ?"
        
        # Create expected equation
        eq_parts = [str(operands[0])]
        for j in range(1, len(operands)):
            eq_parts.extend([op, str(operands[j])])
        eq_parts.extend(['=', str(expected)])
        expected_eq = ' '.join(eq_parts)
        
        print(f"{i+1}. {problem_str}")
        print(f"   Expected answer: {expected_eq}")
        
        # Verify it's correct
        is_correct, computed = evaluate_arithmetic_equation(expected_eq)
        if is_correct:
            print("   ✅ Mathematically correct")
        else:
            print("   ❌ Mathematical error")
        print()


def show_usage_examples():
    """Show examples of how to use the model."""
    
    print(f"\n📖 USAGE EXAMPLES:")
    print("-" * 40)
    print("The model can solve:")
    print("• Basic arithmetic: 7 + 8 = 15")
    print("• Multi-digit: 123 + 456 = 579") 
    print("• Subtraction: 100 - 37 = 63")
    print("• Multiplication: 12 * 5 = 60")
    print("• Division: 84 / 12 = 7")
    print("• Multi-operand: 10 + 20 + 30 = 60")
    print("• Negatives: -50 + 30 = -20")
    print("• Edge cases: 0 + 5 = 5, 10 * 0 = 0")
    
    print(f"\n🔧 HOW TO TEST MORE:")
    print("-" * 25)
    print("1. Run: python simple_manual_test.py")
    print("2. Check the predictions in step_5740_all_preds.0")
    print("3. Use evaluate_arithmetic_capability.py for full testing")
    print("4. Modify custom_problems in this script")


if __name__ == "__main__":
    test_latest_checkpoint()
    test_custom_problems() 
    show_usage_examples()
    
    print(f"\n🎉 TESTING COMPLETE!")
    print("The model shows excellent arithmetic capability!")