from typing import Optional, List
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/arithmetic-1k-aug-1000"
    
    # Problem generation
    num_problems: int = 1000
    max_digits: int = 4
    operations: List[str] = ["+", "-", "*", "/"]
    
    # Data augmentation
    num_aug: int = 1000
    
    # Difficulty control
    include_division: bool = True
    include_negative: bool = True
    max_operands: int = 2  # Number of operands (2 for binary operations)


def generate_arithmetic_problem(max_digits: int, operations: List[str], include_negative: bool, max_operands: int):
    """Generate a single arithmetic problem and its solution."""
    
    # Choose operation
    op = np.random.choice(operations)
    
    # Generate operands
    operands = []
    for _ in range(max_operands):
        # Generate number within digit range
        if max_digits == 1:
            num = np.random.randint(1, 10)
        else:
            num = np.random.randint(10**(max_digits-1), 10**max_digits)
        
        # Add negative sign with probability
        if include_negative and np.random.rand() < 0.3:
            num = -num
            
        operands.append(num)
    
    # Initialize result
    result = 0
    
    # Ensure valid division (no division by zero, integer result)
    if op == "/" and len(operands) == 2:
        a, b = operands
        if b == 0:
            b = np.random.randint(1, 10)
        # Make sure division results in integer
        result = a * b
        operands = [result, b]
    
    # Calculate result
    if op == "+":
        result = sum(operands)
    elif op == "-":
        result = operands[0] - operands[1]
    elif op == "*":
        result = operands[0] * operands[1]
    elif op == "/" and len(operands) == 2:
        result = operands[0] // operands[1]
    
    return operands, op, result


def arithmetic_to_sequence(operands: List[int], operation: str, result: int):
    """Convert arithmetic problem to token sequence.
    
    Token mapping:
    0: PAD
    1: digits 0-9 (1-10)
    11: +
    12: -  
    13: *
    14: /
    15: =
    16: EOS
    17: negative sign (-)
    """
    
    def number_to_tokens(num: int) -> List[int]:
        """Convert number to list of digit tokens."""
        if num < 0:
            tokens = [17]  # negative sign
            num = abs(num)
        else:
            tokens = []
        
        # Convert digits (0-9 mapped to tokens 1-10)
        digits = [int(d) for d in str(num)]
        tokens.extend([d + 1 for d in digits])
        return tokens
    
    tokens = []
    
    # Add operands with operation
    for i, operand in enumerate(operands):
        if i > 0:
            # Add operation token
            op_token = {"+": 11, "-": 12, "*": 13, "/": 14}[operation]
            tokens.append(op_token)
        
        tokens.extend(number_to_tokens(operand))
    
    # Add equals and result
    tokens.append(15)  # =
    tokens.extend(number_to_tokens(result))
    tokens.append(16)  # EOS
    
    return tokens


def augment_problem(operands: List[int], operation: str, result: int):
    """Apply simple augmentations to arithmetic problems."""
    
    # Augmentation 1: Swap operands for commutative operations
    if operation in ["+", "*"] and len(operands) == 2:
        if np.random.rand() < 0.5:
            operands = [operands[1], operands[0]]
    
    # Augmentation 2: Add/subtract same value to maintain equality
    if operation in ["+", "-"] and np.random.rand() < 0.3:
        delta = np.random.randint(-10, 11)
        if operation == "+":
            # (a + b) = c -> (a + delta) + (b - delta) = c
            operands = [operands[0] + delta, operands[1] - delta]
        elif operation == "-":
            # (a - b) = c -> (a + delta) - (b + delta) = c
            operands = [operands[0] + delta, operands[1] + delta]
    
    return operands, operation, result


def convert_subset(set_name: str, config: DataProcessConfig):
    """Generate arithmetic problems for given dataset split."""
    
    # Adjust problem count for train/test
    num_problems = config.num_problems if set_name == "train" else config.num_problems // 5
    num_augments = config.num_aug if set_name == "train" else 0
    
    operations = config.operations.copy()
    if not config.include_division:
        operations = [op for op in operations if op != "/"]
    
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    # Generate problems
    for _ in tqdm(range(num_problems), desc=f"Generating {set_name} problems"):
        # Generate base problem
        operands, op, result = generate_arithmetic_problem(
            config.max_digits, operations, config.include_negative, config.max_operands
        )
        
        # Create augmentations
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                # Original problem
                aug_operands, aug_op, aug_result = operands, op, result
            else:
                # Apply augmentation
                aug_operands, aug_op, aug_result = augment_problem(operands, op, result)
            
            # Convert to sequence
            input_seq = arithmetic_to_sequence(aug_operands, aug_op, aug_result)
            
            # For this task, input and label are the same (model learns to complete the equation)
            # We can mask out the result part in labels during training
            label_seq = input_seq.copy()
            
            results["inputs"].append(input_seq)
            results["labels"].append(label_seq) 
            
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)  # All arithmetic problems have same identifier
        
        # Push group
        results["group_indices"].append(puzzle_id)
    
    # Pad sequences to fixed length
    max_seq_len = 32  # Should be enough for most arithmetic problems
    
    def pad_sequence(seq, target_len):
        if len(seq) > target_len:
            return seq[:target_len]
        return seq + [0] * (target_len - len(seq))
    
    # Convert to numpy arrays with padding
    results["inputs"] = np.array([pad_sequence(seq, max_seq_len) for seq in results["inputs"]], dtype=np.int32)
    results["labels"] = np.array([pad_sequence(seq, max_seq_len) for seq in results["labels"]], dtype=np.int32)
    
    results["group_indices"] = np.array(results["group_indices"], dtype=np.int32)
    results["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32) 
    results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"], dtype=np.int32)
    
    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=max_seq_len,
        vocab_size=18,  # 0=PAD, 1-10=digits, 11-15=ops, 16=EOS, 17=neg
        
        pad_id=0,
        ignore_label_id=0,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1 + (num_augments if set_name == "train" else 0),
        sets=["all"]
    )
    
    # Save data
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    # Save arrays
    for set_name_inner in ["all"]:  # Only one set for arithmetic
        for field_name, field_data in results.items():
            np.save(os.path.join(save_dir, f"{set_name_inner}__{field_name}.npy"), field_data)
    
    print(f"Saved {set_name} dataset with {len(results['inputs'])} examples")
    print(f"Example problem: {results['inputs'][0]}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Build arithmetic dataset for training HRM."""
    
    print(f"Building arithmetic dataset with config: {config}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate train and test sets
    for split in ["train", "test"]:
        convert_subset(split, config)
    
    print(f"Dataset saved to: {config.output_dir}")
    print("Token mapping:")
    print("0: PAD, 1-10: digits 0-9, 11: +, 12: -, 13: *, 14: /, 15: =, 16: EOS, 17: negative")


if __name__ == "__main__":
    cli()