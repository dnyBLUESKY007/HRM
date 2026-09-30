from typing import Optional, List
import os
import json
import numpy as np
import random

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class EnhancedDataProcessConfig(BaseModel):
    output_dir: str = "data/arithmetic-enhanced-mixed"
    
    # Problem generation
    num_problems: int = 500  # More problems for better coverage
    num_aug: int = 100  # Fewer augmentations per problem
    
    # Operation distribution (will be balanced)
    operations: List[str] = ["+", "-", "*", "/"]
    
    # Difficulty levels for diverse results
    easy_problems: int = 40   # 1-2 digits, 2 operands
    medium_problems: int = 40  # 3-4 digits, 2-3 operands  
    hard_problems: int = 20   # 4-6 digits, 2-4 operands
    
    # Control settings
    include_negative: bool = True
    max_operands: int = 4


def generate_balanced_arithmetic_problem(config: EnhancedDataProcessConfig, difficulty: str = "medium"):
    """Generate arithmetic problems with balanced operations and meaningful results."""
    
    # Choose operation with balanced distribution
    operation = random.choice(config.operations)
    
    # Set parameters based on difficulty
    if difficulty == "easy":
        max_digits = random.randint(1, 2)
        num_operands = random.randint(2, 2)  # Only 2 operands for easy
    elif difficulty == "medium":
        max_digits = random.randint(3, 4)  
        num_operands = random.randint(2, 3)
    else:  # hard
        max_digits = random.randint(4, 6)
        num_operands = random.randint(2, min(4, config.max_operands))
    
    # Generate operands with constraints for meaningful results
    operands = []
    
    for i in range(num_operands):
        if max_digits == 1:
            num = random.randint(1, 9)
        elif max_digits == 2:
            num = random.randint(10, 99)
        elif max_digits == 3:
            num = random.randint(100, 999)
        elif max_digits == 4:
            num = random.randint(1000, 9999)
        elif max_digits == 5:
            num = random.randint(10000, 99999)
        else:  # 6 digits
            num = random.randint(100000, 999999)
        
        # Apply negative with controlled probability
        if config.include_negative and random.random() < 0.2:  # 20% chance
            num = -num
            
        operands.append(num)
    
    # Special handling for division to ensure integer results
    if operation == "/":
        # Create division problems that result in integers
        if num_operands == 2:
            # For a / b = c, we set a = b * c where c is chosen
            divisor = operands[1]
            if divisor == 0:
                divisor = random.randint(1, 10)
            
            # Choose a reasonable quotient
            if difficulty == "easy":
                quotient = random.randint(1, 10)
            elif difficulty == "medium": 
                quotient = random.randint(1, 100)
            else:
                quotient = random.randint(1, 1000)
                
            dividend = divisor * quotient
            operands = [dividend, divisor]
        else:
            # For multiple division, ensure each step gives integer result
            # Simplify to 2 operands for now
            operands = operands[:2]
            divisor = operands[1]
            if divisor == 0:
                divisor = random.randint(1, 10)
            quotient = random.randint(1, 20)
            dividend = divisor * quotient
            operands = [dividend, divisor]
    
    # Special handling for multiplication to avoid overflow
    elif operation == "*":
        if num_operands > 2 or max_digits > 4:
            # Use smaller numbers for multiplication to avoid huge results
            operands = [random.randint(2, 50) for _ in range(min(num_operands, 3))]
            if config.include_negative and random.random() < 0.3:
                operands[0] = -operands[0]
    
    # Calculate result using Python eval
    if len(operands) >= 2:
        expr_parts = [str(operands[0])]
        for i in range(1, len(operands)):
            expr_parts.append(operation)
            expr_parts.append(str(operands[i]))
        
        expression = ' '.join(expr_parts)
        
        try:
            if operation == "/":
                # Use integer division
                expression = expression.replace('/', '//')
            result = eval(expression)
            
            if isinstance(result, float):
                result = int(result)
                
        except (ZeroDivisionError, ValueError, SyntaxError):
            # Fallback: generate a simple valid problem
            operands = [random.randint(1, 100), random.randint(1, 100)]
            if operation == "+":
                result = operands[0] + operands[1]
            elif operation == "-":
                result = operands[0] - operands[1]  
            elif operation == "*":
                result = operands[0] * operands[1]
            else:  # division
                if operands[1] != 0:
                    result = operands[0] // operands[1]
                else:
                    operands[1] = 1
                    result = operands[0]
    else:
        result = operands[0] if operands else 0
    
    return operands, operation, result


def arithmetic_to_sequence(operands: List[int], operation: str, result: int, max_seq_len: int = 32):
    """Convert arithmetic problem to token sequence."""
    
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
    result_tokens = number_to_tokens(result)
    
    # Check if sequence will fit in max_seq_len
    total_length = len(tokens) + len(result_tokens) + 1  # +1 for EOS
    if total_length > max_seq_len:
        return None
    
    tokens.extend(result_tokens)
    tokens.append(16)  # EOS
    
    return tokens


def augment_balanced_problem(operands: List[int], operation: str, result: int):
    """Apply balanced augmentations that maintain correctness."""
    
    # Simple augmentations that preserve mathematical correctness
    if operation in ["+", "*"] and len(operands) == 2:
        # Commutative operations: swap operands
        if random.random() < 0.5:
            operands = [operands[1], operands[0]]
    
    # For addition/subtraction, apply value-preserving transformations occasionally
    if operation in ["+", "-"] and len(operands) == 2 and random.random() < 0.2:
        delta = random.randint(-5, 5)  # Smaller delta to avoid huge numbers
        if operation == "+":
            # (a + b) = c -> (a + delta) + (b - delta) = c
            operands = [operands[0] + delta, operands[1] - delta]
        elif operation == "-":
            # (a - b) = c -> (a + delta) - (b + delta) = c  
            operands = [operands[0] + delta, operands[1] + delta]
    
    # Recalculate result to ensure consistency
    if len(operands) >= 2:
        expr_parts = [str(operands[0])]
        for i in range(1, len(operands)):
            expr_parts.append(operation)
            expr_parts.append(str(operands[i]))
        
        expression = ' '.join(expr_parts)
        try:
            if operation == "/":
                expression = expression.replace('/', '//')
            result = eval(expression)
            if isinstance(result, float):
                result = int(result)
        except:
            pass  # Keep original result if evaluation fails
    
    return operands, operation, result


def create_test_problems():
    """Create a diverse test set with known difficulty levels."""
    
    test_cases = [
        # Easy addition (1-2 digits)
        ([7, 8], "+", 15),
        ([23, 45], "+", 68),
        ([9, 17], "+", 26),
        
        # Easy subtraction  
        ([50, 23], "-", 27),
        ([100, 37], "-", 63),
        ([88, 19], "-", 69),
        
        # Easy multiplication
        ([6, 7], "*", 42),
        ([9, 8], "*", 72),
        ([12, 5], "*", 60),
        
        # Easy division
        ([84, 12], "/", 7),
        ([96, 8], "/", 12),
        ([63, 9], "/", 7),
        
        # Medium problems (3-4 digits, 2-3 operands)
        ([456, 123, 789], "+", 1368),
        ([1000, 234, 156], "-", 610),
        ([25, 16, 3], "*", 1200),
        
        # Hard problems with multiple operands
        ([1000, 200, 300, 100], "+", 1600),
        ([2000, 500, 250, 50], "-", 1200),
        ([5, 4, 3, 2], "*", 120),
        
        # Mixed positive/negative
        ([-50, 30], "+", -20),
        ([100, -25], "-", 125),
        ([-8, 6], "*", -48),
        ([-144, -12], "/", 12),
        
        # Edge cases
        ([0, 5], "+", 5),
        ([10, 0], "*", 0),
        ([1, 1], "*", 1),
    ]
    
    return test_cases


def convert_subset(set_name: str, config: EnhancedDataProcessConfig):
    """Generate enhanced arithmetic problems."""
    
    # Calculate problem distribution
    is_train = (set_name == "train")
    
    if is_train:
        total_problems = config.num_problems
        num_augments = config.num_aug
        
        # Distribution by difficulty
        easy_count = int(total_problems * config.easy_problems / 100)
        medium_count = int(total_problems * config.medium_problems / 100) 
        hard_count = total_problems - easy_count - medium_count
        
        difficulties = (["easy"] * easy_count + 
                       ["medium"] * medium_count + 
                       ["hard"] * hard_count)
        random.shuffle(difficulties)
    else:
        # Test set: use predefined meaningful problems
        test_problems = create_test_problems()
        total_problems = len(test_problems)
        num_augments = 0
        difficulties = ["test"] * total_problems
    
    max_seq_len = 32
    
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    # Generate problems
    problems_desc = f"Generating {set_name} problems"
    for prob_idx in tqdm(range(total_problems), desc=problems_desc):
        
        if is_train:
            difficulty = difficulties[prob_idx]
            operands, op, result = generate_balanced_arithmetic_problem(config, difficulty)
        else:
            # Use predefined test problems
            operands, op, result = test_problems[prob_idx]
        
        examples_added_this_group = 0
        
        # Create augmentations for training
        augmentation_range = range(1 + (num_augments if is_train else 0))
        
        for aug_idx in augmentation_range:
            if aug_idx == 0:
                # Original problem
                aug_operands, aug_op, aug_result = operands, op, result
            else:
                # Apply augmentation
                aug_operands, aug_op, aug_result = augment_balanced_problem(operands, op, result)
            
            # Convert to sequence
            input_seq = arithmetic_to_sequence(aug_operands, aug_op, aug_result, max_seq_len)
            
            # Skip if sequence is too long
            if input_seq is None:
                continue
                
            label_seq = input_seq.copy()
            
            results["inputs"].append(input_seq)
            results["labels"].append(label_seq)
            
            example_id += 1
            puzzle_id += 1
            examples_added_this_group += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
        
        # Only push group if we added examples
        if examples_added_this_group > 0:
            results["group_indices"].append(puzzle_id)
    
    # Pad sequences to fixed length
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
        mean_puzzle_examples=1 + (num_augments if is_train else 0),
        sets=["all"]
    )
    
    # Save data
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    # Save arrays
    for set_name_inner in ["all"]:
        for field_name, field_data in results.items():
            np.save(os.path.join(save_dir, f"{set_name_inner}__{field_name}.npy"), field_data)
    
    print(f"Saved {set_name} dataset with {len(results['inputs'])} examples")
    
    # Show some statistics
    if len(results['inputs']) > 0:
        print(f"Example problems:")
        for i in range(min(3, len(results['inputs']))):
            # Decode sequence to show what was generated
            tokens = results['inputs'][i]
            
            # Simple decoding for display
            token_to_char = {
                0: '', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 
                6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
                11: ' + ', 12: ' - ', 13: ' * ', 14: ' / ', 15: ' = ', 16: '', 17: '-'
            }
            
            display = ""
            for token in tokens:
                if token == 0:  # PAD
                    break
                if token == 16:  # EOS
                    break
                display += token_to_char.get(token, f'[{token}]')
            
            print(f"  {display}")


@cli.command(singleton=True)
def preprocess_data(config: EnhancedDataProcessConfig):
    """Build enhanced arithmetic dataset with balanced operations and meaningful results."""
    
    print(f"Building enhanced arithmetic dataset with config:")
    print(f"  Output: {config.output_dir}")
    print(f"  Problems: {config.num_problems} (Easy: {config.easy_problems}%, Medium: {config.medium_problems}%, Hard: {config.hard_problems}%)")
    print(f"  Augmentations: {config.num_aug}")
    print(f"  Operations: {config.operations}")
    print(f"  Max operands: {config.max_operands}")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate train and test sets
    for split in ["train", "test"]:
        convert_subset(split, config)
    
    print(f"\nDataset saved to: {config.output_dir}")
    print("Token mapping:")
    print("0: PAD, 1-10: digits 0-9, 11: +, 12: -, 13: *, 14: /, 15: =, 16: EOS, 17: negative")


if __name__ == "__main__":
    cli()