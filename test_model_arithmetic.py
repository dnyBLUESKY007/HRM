#!/usr/bin/env python3
"""
Test script to evaluate model's arithmetic capabilities with custom examples.
"""

import sys
import os
import torch
import numpy as np
sys.path.append('dataset')

from arithmetic_evaluation import decode_arithmetic_sequence, evaluate_arithmetic_equation
from build_arithmetic_dataset import arithmetic_to_sequence
from pretrain import PretrainConfig, init_train_state, create_dataloader
import yaml


def create_custom_test_examples():
    """Create meaningful arithmetic test cases."""
    
    test_cases = [
        # Simple addition
        ([123, 456], "+", 579),
        ([789, 321], "+", 1110),
        
        # Simple subtraction  
        ([1000, 234], "-", 766),
        ([555, 333], "-", 222),
        
        # Simple multiplication (smaller numbers to avoid overflow)
        ([12, 34], "*", 408),
        ([25, 16], "*", 400),
        
        # Division with integer results
        ([144, 12], "/", 12),
        ([100, 5], "/", 20),
        
        # 3-operand addition
        ([10, 20, 30], "+", 60),
        ([100, 200, 300], "+", 600),
        
        # 3-operand subtraction  
        ([1000, 200, 300], "-", 500),
        ([500, 100, 50], "-", 350),
    ]
    
    return test_cases


def tokens_to_arithmetic_sequence(operands, operation, result):
    """Convert arithmetic problem to token sequence for testing."""
    return arithmetic_to_sequence(operands, operation, result, max_seq_len=32)


def test_model_on_custom_examples(checkpoint_path):
    """Test the model on custom arithmetic examples."""
    
    print("Loading model...")
    
    # Load config
    with open(os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    
    # Create dummy dataloader to get metadata
    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1, 
        global_batch_size=1, rank=0, world_size=1
    )
    
    # Initialize model
    train_state = init_train_state(config, train_metadata, world_size=1)
    
    # Load checkpoint
    try:
        train_state.model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(checkpoint_path, map_location="cuda").items()}, assign=True)
    
    train_state.model.eval()
    
    print("Testing model on custom arithmetic examples...")
    print("=" * 60)
    
    # Create test cases
    test_cases = create_custom_test_examples()
    
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for i, (operands, operation, expected_result) in enumerate(test_cases):
            # Convert to token sequence
            input_tokens = tokens_to_arithmetic_sequence(operands, operation, expected_result)
            
            if input_tokens is None:
                print(f"Test {i+1}: Sequence too long, skipping")
                continue
                
            # Prepare input (mask out the result part for prediction)
            input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).cuda()
            
            # Find the equals sign to mask everything after it
            equals_pos = (input_tokens == 15).nonzero(as_tuple=True)
            if len(equals_pos[1]) > 0:
                mask_start = equals_pos[1][0] + 1  # Start masking after '='
                masked_input = input_tokens.clone()
                masked_input[0, mask_start:] = 0  # Mask with PAD tokens
            else:
                masked_input = input_tokens
            
            # Create batch for model
            batch = {
                'inputs': masked_input,
                'labels': input_tokens,  # Full sequence as label
                'puzzle_identifiers': torch.zeros(1, dtype=torch.long).cuda()
            }
            
            # Run model
            try:
                # Initialize carry
                carry = train_state.model.initial_carry(batch)
                
                # Forward pass
                carry, loss, metrics, outputs, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=["logits"]
                )
                
                # Get predictions
                if "logits" in outputs:
                    pred_tokens = torch.argmax(outputs["logits"], dim=-1)
                    
                    # Decode original and prediction
                    original_eq, original_result = decode_arithmetic_sequence(input_tokens[0])
                    pred_eq, pred_result = decode_arithmetic_sequence(pred_tokens[0])
                    
                    print(f"Test {i+1}: {' '.join(map(str, operands))} {operation} ? = ?")
                    print(f"  Expected: {original_eq}")
                    print(f"  Predicted: {pred_eq}")
                    
                    # Check correctness
                    if pred_eq:
                        is_correct, eval_result = evaluate_arithmetic_equation(pred_eq)
                        print(f"  Mathematically correct: {is_correct}")
                        print(f"  Expected result: {expected_result}, Got: {pred_result}")
                        
                        if is_correct and pred_result == expected_result:
                            correct_count += 1
                            print("  ✓ CORRECT")
                        else:
                            print("  ✗ INCORRECT")
                    else:
                        print("  ✗ FAILED TO DECODE")
                    
                    total_count += 1
                    print()
                    
            except Exception as e:
                print(f"Test {i+1}: Error during inference: {e}")
                print()
    
    print("=" * 60)
    print(f"Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_model_arithmetic.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    test_model_on_custom_examples(checkpoint_path)


if __name__ == "__main__":
    main()