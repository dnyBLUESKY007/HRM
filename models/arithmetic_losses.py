"""
Arithmetic-enhanced loss heads for HRM training.

This module extends the standard ACT loss head with arithmetic-specific 
evaluation metrics and correctness checking.
"""

from typing import Any, Tuple, Dict, Sequence, Optional
import sys
import os

import torch
import torch.nn.functional as F
from torch import nn

# Add dataset path to sys.path to import arithmetic evaluation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

from losses import ACTLossHead, IGNORE_LABEL_ID
try:
    from arithmetic_evaluation import compute_arithmetic_metrics, decode_arithmetic_sequence, evaluate_arithmetic_equation
except ImportError:
    print("Warning: Could not import arithmetic_evaluation. Arithmetic-specific metrics will be disabled.")
    compute_arithmetic_metrics = None
    decode_arithmetic_sequence = None
    evaluate_arithmetic_equation = None


class ArithmeticACTLossHead(ACTLossHead):
    """Enhanced ACT Loss Head with arithmetic-specific evaluation metrics."""
    
    def __init__(self, model: nn.Module, loss_type: str, enable_arithmetic_eval: bool = True):
        super().__init__(model, loss_type)
        self.enable_arithmetic_eval = enable_arithmetic_eval and (compute_arithmetic_metrics is not None)
        
    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # Call parent forward to get standard metrics
        new_carry, loss, metrics, detached_outputs, all_finish = super().forward(
            return_keys=return_keys, **model_kwargs
        )
        
        # Add arithmetic-specific metrics
        if self.enable_arithmetic_eval and not self.training:
            try:
                self._add_arithmetic_metrics(new_carry, detached_outputs, metrics)
            except Exception as e:
                print(f"Warning: Arithmetic evaluation failed: {e}")
        
        return new_carry, loss, metrics, detached_outputs, all_finish
    
    def _add_arithmetic_metrics(self, new_carry, detached_outputs, metrics):
        """Add arithmetic-specific metrics to the metrics dictionary."""
        
        if not hasattr(new_carry, 'current_data') or 'labels' not in new_carry.current_data:
            return
            
        labels = new_carry.current_data["labels"]
        
        # Try to get logits from detached_outputs or from the forward pass
        logits = None
        if "logits" in detached_outputs:
            logits = detached_outputs["logits"]
        
        if logits is not None:
            # Compute arithmetic-specific metrics
            arith_metrics = compute_arithmetic_metrics(logits, labels, ignore_index=IGNORE_LABEL_ID)
            
            # Add to existing metrics with 'arith_' prefix
            for key, value in arith_metrics.items():
                if torch.is_tensor(value):
                    metrics[f"arith_{key}"] = value.detach()
                else:
                    metrics[f"arith_{key}"] = torch.tensor(value, device=labels.device)
    
    def compute_detailed_arithmetic_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute detailed arithmetic accuracy metrics for analysis.
        
        This method provides more detailed breakdown of accuracy metrics
        that can be used for debugging and analysis.
        """
        
        if not self.enable_arithmetic_eval:
            return {}
            
        pred_tokens = torch.argmax(predictions, dim=-1)
        batch_size = predictions.shape[0]
        
        # Detailed tracking
        results = {
            'total_problems': 0,
            'exact_match': 0,
            'equation_correct': 0, 
            'result_only_correct': 0,
            'parse_errors': 0,
            'syntax_errors': 0
        }
        
        for i in range(batch_size):
            results['total_problems'] += 1
            
            # Decode sequences
            pred_eq, pred_result = decode_arithmetic_sequence(pred_tokens[i])
            true_eq, true_result = decode_arithmetic_sequence(labels[i])
            
            if pred_eq is None:
                results['parse_errors'] += 1
                continue
                
            # Check exact token match
            if torch.equal(pred_tokens[i], labels[i]):
                results['exact_match'] += 1
                results['equation_correct'] += 1
                results['result_only_correct'] += 1
                continue
                
            # Check mathematical correctness
            is_math_correct, expected_result = evaluate_arithmetic_equation(pred_eq)
            
            if is_math_correct:
                results['equation_correct'] += 1
                results['result_only_correct'] += 1
            elif pred_result is not None and expected_result is not None:
                # Check if just the result number is correct
                if pred_result == expected_result:
                    results['result_only_correct'] += 1
        
        # Convert to percentages
        if results['total_problems'] > 0:
            for key in ['exact_match', 'equation_correct', 'result_only_correct']:
                results[f"{key}_pct"] = results[key] / results['total_problems'] * 100
        
        return results


def create_arithmetic_enhanced_loss(base_loss_type: str = "softmax_cross_entropy"):
    """Factory function to create arithmetic-enhanced loss head."""
    return lambda model: ArithmeticACTLossHead(model, base_loss_type)