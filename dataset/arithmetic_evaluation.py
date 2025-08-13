"""
Arithmetic-specific evaluation utilities for HRM.

This module provides functions to evaluate arithmetic problem solving performance,
including exact match accuracy and partial correctness checking.
"""

from typing import List, Tuple, Optional, Dict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


def decode_arithmetic_sequence(tokens) -> Tuple[Optional[str], Optional[int]]:
    """Decode a tokenized arithmetic sequence back to readable format.
    
    Token mapping:
    0: PAD, 1-10: digits 0-9, 11: +, 12: -, 13: *, 14: /, 15: =, 16: EOS, 17: negative
    
    Returns:
        Tuple of (equation_string, result_value) or (None, None) if parsing fails
    """
    
    # Token mapping
    token_to_char = {
        0: 'PAD', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 
        6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
        11: '+', 12: '-', 13: '*', 14: '/', 15: '=', 16: 'EOS', 17: 'NEG'
    }
    
    try:
        # Convert to list and stop at PAD or EOS
        if TORCH_AVAILABLE and torch is not None and torch.is_tensor(tokens):
            token_list = tokens.cpu().numpy().tolist()
        elif NUMPY_AVAILABLE and np is not None and isinstance(tokens, np.ndarray):
            token_list = tokens.tolist()
        else:
            token_list = list(tokens)  # Assume it's already a list or iterable
        
        # Find end of sequence (PAD or EOS)
        end_idx = len(token_list)
        for i, token in enumerate(token_list):
            if token == 0 or token == 16:  # PAD or EOS
                end_idx = i
                break
        
        # Convert tokens to string components
        components = []
        i = 0
        while i < end_idx:
            token = token_list[i]
            
            if token == 17:  # NEG
                # Start of a negative number
                num_str = '-'
                i += 1
                # Collect digits
                while i < end_idx and 1 <= token_list[i] <= 10:
                    num_str += token_to_char[token_list[i]]
                    i += 1
                components.append(num_str)
            elif 1 <= token <= 10:  # Digit
                # Start of a positive number
                num_str = token_to_char[token]
                i += 1
                # Collect more digits
                while i < end_idx and 1 <= token_list[i] <= 10:
                    num_str += token_to_char[token_list[i]]
                    i += 1
                components.append(num_str)
            elif token in [11, 12, 13, 14, 15]:  # Operators and =
                components.append(token_to_char[token])
                i += 1
            else:
                i += 1
        
        # Join components
        equation_str = ' '.join(components)
        
        # Extract result (everything after =)
        if '=' in equation_str:
            parts = equation_str.split('=')
            if len(parts) >= 2:
                result_str = parts[1].strip()
                try:
                    result_value = int(result_str)
                    return equation_str, result_value
                except ValueError:
                    pass
        
        return equation_str, None
        
    except Exception:
        return None, None


def evaluate_arithmetic_equation(equation_str: str) -> Tuple[bool, Optional[int]]:
    """Evaluate an arithmetic equation string and check correctness.
    
    Args:
        equation_str: String like "1234 + 5678 = 6912"
    
    Returns:
        Tuple of (is_correct, expected_result)
    """
    
    try:
        if '=' not in equation_str:
            return False, None
            
        parts = equation_str.split('=')
        if len(parts) != 2:
            return False, None
            
        expression = parts[0].strip()
        predicted_result_str = parts[1].strip()
        
        try:
            predicted_result = int(predicted_result_str)
        except ValueError:
            return False, None
            
        # Safely evaluate the expression
        # Replace operators with Python equivalents
        expression = expression.replace('/', '//')  # Integer division
        
        # Validate that expression only contains safe characters
        allowed_chars = set('0123456789+-*/ ()')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return False, None
            
        try:
            expected_result = eval(expression)
            is_correct = (predicted_result == expected_result)
            return is_correct, expected_result
        except:
            return False, None
            
    except Exception:
        return False, None


def compute_arithmetic_metrics(predictions, labels, 
                             ignore_index: int = -100) -> Dict[str, any]:
    """Compute arithmetic-specific evaluation metrics.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, vocab_size] 
        labels: Ground truth labels [batch_size, seq_len]
        ignore_index: Index to ignore in evaluation
    
    Returns:
        Dictionary of metric tensors
    """
    
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch is required for compute_arithmetic_metrics")
    
    device = predictions.device
    batch_size = predictions.shape[0]
    
    # Get predicted tokens
    pred_tokens = torch.argmax(predictions, dim=-1)  # [batch_size, seq_len]
    
    # Compute standard token-level accuracy
    valid_mask = (labels != ignore_index)
    token_correct = (pred_tokens == labels) & valid_mask
    
    # Token-level metrics
    total_valid_tokens = valid_mask.sum()
    correct_tokens = token_correct.sum()
    token_accuracy = correct_tokens.float() / total_valid_tokens.clamp_min(1)
    
    # Sequence-level exact match
    seq_correct = (token_correct.sum(dim=1) == valid_mask.sum(dim=1))
    exact_accuracy = seq_correct.sum().float() / batch_size
    
    # Arithmetic-specific metrics
    equation_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
    partial_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
    result_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # Decode predicted and true sequences
        pred_eq, pred_result = decode_arithmetic_sequence(pred_tokens[i])
        true_eq, true_result = decode_arithmetic_sequence(labels[i])
        
        if pred_eq is not None and true_eq is not None:
            # Check if equation is mathematically correct
            is_math_correct, expected_result = evaluate_arithmetic_equation(pred_eq)
            equation_correct[i] = is_math_correct
            
            # Check if just the numerical result is correct
            if pred_result is not None and expected_result is not None:
                result_correct[i] = (pred_result == expected_result)
                
            # Partial correctness: correct digits in result
            if pred_result is not None and true_result is not None:
                pred_digits = str(abs(pred_result))
                true_digits = str(abs(true_result))
                # Check if at least some digits match
                partial_correct[i] = any(d in true_digits for d in pred_digits)
    
    return {
        'token_accuracy': token_accuracy,
        'exact_match_accuracy': exact_accuracy,
        'equation_correctness': equation_correct.sum().float() / batch_size,
        'result_correctness': result_correct.sum().float() / batch_size,
        'partial_correctness': partial_correct.sum().float() / batch_size,
        'total_examples': torch.tensor(batch_size, dtype=torch.float, device=device),
        'equation_correct_count': equation_correct.sum().float(),
        'result_correct_count': result_correct.sum().float(),
    }


def print_arithmetic_examples(predictions, labels, 
                            num_examples: int = 5) -> None:
    """Print example predictions for debugging.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len] 
        num_examples: Number of examples to print
    """
    
    pred_tokens = torch.argmax(predictions, dim=-1)
    batch_size = min(num_examples, predictions.shape[0])
    
    print(f"\n=== Arithmetic Examples ===")
    for i in range(batch_size):
        pred_eq, pred_result = decode_arithmetic_sequence(pred_tokens[i])
        true_eq, true_result = decode_arithmetic_sequence(labels[i])
        
        print(f"\nExample {i+1}:")
        print(f"  Predicted: {pred_eq}")
        print(f"  True:      {true_eq}")
        
        if pred_eq and true_eq:
            is_correct, expected = evaluate_arithmetic_equation(pred_eq)
            print(f"  Correct:   {is_correct} (expected: {expected})")


def create_arithmetic_loss_head(base_loss_head_class):
    """Create an enhanced loss head with arithmetic-specific metrics."""
    
    class ArithmeticACTLossHead(base_loss_head_class):
        def forward(self, return_keys, **model_kwargs):
            # Call parent forward
            new_carry, loss, metrics, detached_outputs, all_finish = super().forward(
                return_keys=return_keys, **model_kwargs
            )
            
            # Add arithmetic-specific metrics if we have predictions
            if "logits" in detached_outputs or hasattr(new_carry, 'current_data'):
                try:
                    logits = detached_outputs.get("logits")
                    if logits is None and hasattr(new_carry, 'current_data'):
                        # Try to get logits from model outputs (this might need adjustment based on actual model structure)
                        pass
                    
                    labels = new_carry.current_data["labels"]
                    
                    if logits is not None:
                        # Compute arithmetic-specific metrics
                        arith_metrics = compute_arithmetic_metrics(logits, labels)
                        
                        # Add to existing metrics with prefix
                        for key, value in arith_metrics.items():
                            metrics[f"arith_{key}"] = value
                        
                except Exception as e:
                    # Fallback gracefully if arithmetic evaluation fails
                    print(f"Warning: Arithmetic evaluation failed: {e}")
                    pass
                    
            return new_carry, loss, metrics, detached_outputs, all_finish
    
    return ArithmeticACTLossHead