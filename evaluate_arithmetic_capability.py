#!/usr/bin/env python3
"""
Arithmetic capability evaluation script.
Provides multiple ways to test a model's arithmetic performance.
"""

import sys
import os
import argparse
import numpy as np
import torch
sys.path.append('dataset')

from arithmetic_evaluation import decode_arithmetic_sequence, evaluate_arithmetic_equation


def quick_dataset_check(dataset_path):
    """Quick check of dataset quality."""
    
    print(f"QUICK DATASET CHECK: {dataset_path}")
    print("="*50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Load test set
    test_dir = os.path.join(dataset_path, 'test')
    if not os.path.exists(test_dir):
        print("❌ No test directory found")
        return False
    
    try:
        test_inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'))
        print(f"✅ Found {len(test_inputs)} test examples")
        
        # Check first few examples
        meaningful_problems = 0
        all_correct = 0
        
        for i in range(min(10, len(test_inputs))):
            eq, result = decode_arithmetic_sequence(test_inputs[i])
            if eq:
                is_correct, expected = evaluate_arithmetic_equation(eq)
                if is_correct:
                    all_correct += 1
                    
                # Check if problem is meaningful (not trivial like x/y=0)
                if expected not in [0] and abs(expected) > 1:
                    meaningful_problems += 1
                    
                if i < 3:  # Show first 3
                    print(f"  Example {i+1}: {eq} -> Expected: {expected}, Correct: {is_correct}")
        
        if meaningful_problems >= 5:
            print(f"✅ Dataset has {meaningful_problems}/10 meaningful problems")
        else:
            print(f"⚠️  Dataset has only {meaningful_problems}/10 meaningful problems")
            
        if all_correct == min(10, len(test_inputs)):
            print("✅ All sampled problems are mathematically consistent")
        else:
            print(f"❌ Only {all_correct}/{min(10, len(test_inputs))} problems are consistent")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False


def evaluate_with_builtin_script(checkpoint_path):
    """Use the built-in evaluation script."""
    
    print("USING BUILT-IN EVALUATION SCRIPT")
    print("="*50)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print("Running: python evaluate.py checkpoint=\"{checkpoint_path}\"")
    print()
    
    # Run evaluation
    import subprocess
    try:
        result = subprocess.run([
            'python', 'evaluate.py', f'checkpoint={checkpoint_path}'
        ], capture_output=True, text=True, timeout=120)
        
        print("EVALUATION RESULTS:")
        print("-" * 20)
        print(result.stdout)
        
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out (>2 minutes)")
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")


def analyze_saved_predictions(checkpoint_dir):
    """Analyze saved prediction files."""
    
    print("ANALYZING SAVED PREDICTIONS")
    print("="*50)
    
    # Look for prediction files
    pred_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_all_preds.0')]
    
    if not pred_files:
        print("❌ No prediction files found")
        return
    
    # Use the latest prediction file
    pred_file = sorted(pred_files)[-1]
    pred_path = os.path.join(checkpoint_dir, pred_file)
    
    print(f"Loading predictions from: {pred_file}")
    
    try:
        predictions = torch.load(pred_path, map_location='cpu')
        
        if 'logits' not in predictions:
            print("❌ No logits found in predictions")
            return
            
        logits = predictions['logits']
        pred_tokens = torch.argmax(logits, dim=-1)
        
        inputs = predictions.get('inputs', None)
        labels = predictions.get('labels', None)
        
        print(f"Analyzing {len(pred_tokens)} predictions...")
        
        # Analyze predictions
        correct_exact = 0
        correct_math = 0
        correct_result = 0
        total_valid = 0
        
        print("\\nSample predictions:")
        for i in range(min(5, len(pred_tokens))):
            
            # Decode input, label, and prediction
            if inputs is not None:
                input_eq, _ = decode_arithmetic_sequence(inputs[i])
            else:
                input_eq = "N/A"
                
            if labels is not None:
                label_eq, label_result = decode_arithmetic_sequence(labels[i])
            else:
                label_eq, label_result = "N/A", None
                
            pred_eq, pred_result = decode_arithmetic_sequence(pred_tokens[i])
            
            print(f"\\n  Example {i+1}:")
            print(f"    Input:      {input_eq}")
            print(f"    Label:      {label_eq}")
            print(f"    Prediction: {pred_eq}")
            
            if pred_eq and label_eq != "N/A":
                total_valid += 1
                
                # Check exact match
                if torch.equal(pred_tokens[i], labels[i]):
                    correct_exact += 1
                    print("    ✅ Exact token match")
                else:
                    print("    ❌ No exact token match")
                
                # Check mathematical correctness
                is_math_correct, expected = evaluate_arithmetic_equation(pred_eq)
                if is_math_correct:
                    correct_math += 1
                    print("    ✅ Mathematically correct")
                else:
                    print(f"    ❌ Math incorrect (expected: {expected})")
                
                # Check result correctness
                if pred_result == label_result and pred_result is not None:
                    correct_result += 1
                    print("    ✅ Correct numerical result")
                else:
                    print(f"    ❌ Wrong result (got: {pred_result}, expected: {label_result})")
        
        # Summary
        print(f"\\nSUMMARY (from {total_valid} valid predictions):")
        if total_valid > 0:
            print(f"  Exact token matches: {correct_exact}/{total_valid} ({correct_exact/total_valid*100:.1f}%)")
            print(f"  Mathematically correct: {correct_math}/{total_valid} ({correct_math/total_valid*100:.1f}%)")
            print(f"  Correct numerical results: {correct_result}/{total_valid} ({correct_result/total_valid*100:.1f}%)")
        else:
            print("  No valid predictions to analyze")
            
    except Exception as e:
        print(f"❌ Error analyzing predictions: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate arithmetic capability of trained models")
    parser.add_argument("--dataset", default="data/arithmetic-enhanced-balanced", 
                       help="Dataset path to check")
    parser.add_argument("--checkpoint", 
                       help="Checkpoint path to evaluate")
    parser.add_argument("--checkpoint-dir",
                       help="Directory containing checkpoints and predictions to analyze")
    parser.add_argument("--quick-check", action="store_true",
                       help="Only run quick dataset check")
    
    args = parser.parse_args()
    
    print("ARITHMETIC CAPABILITY EVALUATION")
    print("="*60)
    print()
    
    # 1. Quick dataset check
    dataset_ok = quick_dataset_check(args.dataset)
    print()
    
    if args.quick_check:
        return
    
    if not dataset_ok:
        print("❌ Dataset check failed. Please fix dataset before evaluating models.")
        return
    
    # 2. Built-in evaluation (if checkpoint provided)
    if args.checkpoint:
        evaluate_with_builtin_script(args.checkpoint)
        print()
    
    # 3. Analyze saved predictions (if checkpoint dir provided)
    if args.checkpoint_dir:
        analyze_saved_predictions(args.checkpoint_dir)
        print()
    
    # 4. Suggestions for improvement
    print("TESTING RECOMMENDATIONS")
    print("="*30)
    print("1. Use --dataset to check dataset quality")
    print("2. Use --checkpoint to run full model evaluation")
    print("3. Use --checkpoint-dir to analyze saved predictions")
    print("4. Train a model on the enhanced dataset for better results")
    print()
    print("Example usage:")
    print("  python evaluate_arithmetic_capability.py --dataset data/arithmetic-enhanced-balanced")
    print("  python evaluate_arithmetic_capability.py --checkpoint path/to/checkpoint")
    print("  python evaluate_arithmetic_capability.py --checkpoint-dir path/to/checkpoint/directory")


if __name__ == "__main__":
    main()