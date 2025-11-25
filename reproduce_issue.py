
import torch
import numpy as np
from typing import Tuple, Union

def numeric_to_string(
    name: Union[Tuple[int, int, int, int], Tuple[int, int, int, int, int]],
) -> str:
    name = [chr(c + 32) for c in name if c != 0]
    name = "".join(name)
    return name

def test_numeric_to_string():
    print("Testing numeric_to_string with tensor...")
    t = torch.tensor([33, 34, 0, 0, 0])
    try:
        s = numeric_to_string(t)
        print(f"Result: {s}")
    except Exception as e:
        print(f"Error: {e}")

def test_numpy_array_creation():
    print("Testing numpy array creation with tensors...")
    Residue = [
        ("name", np.dtype("<U5")),
        ("res_type", np.dtype("i1")),
        ("is_standard", np.dtype("?")),
    ]
    
    res_type = torch.tensor(1)
    is_standard = torch.tensor(True)
    
    res_data = [
        ("ALA", res_type, is_standard)
    ]
    
    try:
        arr = np.array(res_data, dtype=Residue)
        print("Array created successfully")
        print(arr)
    except Exception as e:
        print(f"Error creating array: {e}")

    print("Testing numpy array creation with tensor of size 1 (1-d)...")
    res_type_1d = torch.tensor([1])
    is_standard_1d = torch.tensor([True])
    
    res_data_1d = [
        ("ALA", res_type_1d, is_standard_1d)
    ]
    
    try:
        arr = np.array(res_data_1d, dtype=Residue)
        print("Array created successfully")
        print(arr)
    except Exception as e:
        print(f"Error creating array 1d: {e}")

if __name__ == "__main__":
    test_numeric_to_string()
    test_numpy_array_creation()
