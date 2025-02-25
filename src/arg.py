from dataclasses import dataclass, field
from typing import Optional, List

# Set a arguments to define all using need setting 

# For dataset
@dataclass
class DataArguments:
    root_dir: str = field(
        default="/workspace/Jeming/data/",
        metadata={"help": "root_dir"},
    )
    check_models: Optional[List[str]] = field(
        default_factory=lambda: ["ADC", "T2_FS", "V"],
        metadata={"help": "check_models"},
    )
    model_scale: Optional[List[List[int]]] = field(
        default_factory=lambda: [[0,6000],[0,4500],[0,4500]],
        metadata={"help": "model_scale"},
    )
    target_size: Optional[List[int]] = field(
        default_factory=lambda: [128, 128, 64],
        metadata={"help": "check_models"},
    )
    train_ratio: Optional[float] = field(
        default = 0.7,
        metadata={"help": "train_ratio"},
    )
    val_ratio: Optional[float] = field(
        default = 0.1,
        metadata={"help": "val_ratio"},
    )
    test_ratio: Optional[float] = field(
        default = 0.2,
        metadata={"help": "test_ratio"},
    )

@dataclass
class ModelArguments:
    in_chans: Optional[int] = field(default=4, metadata={"help": "in_chans"})
    out_chans: Optional[int] = field(default=3, metadata={"help": "out_chans"})
    d_state: Optional[int] = field(default=16, metadata={"help": "d_state"})
    d_conv: Optional[int] = field(default=4, metadata={"help": "d_conv"})
    expand: Optional[int] = field(default=2, metadata={"help": "expand"})
    hidden_size: Optional[int] = field(default=768, metadata={"help": "hidden_size"})
    fussion: Optional[List[int]] = field(
        default_factory = lambda: [1, 2, 4, 8],
        metadata={"help": "fussion"},
    )
    kernel_sizes: Optional[List[int]] = field(
        default_factory = lambda: [4, 2, 2, 2],
        metadata={"help": "kernel_sizes"},
    )
    depths: Optional[List[int]] = field(
        default_factory = lambda: [1, 1, 1, 1],
        metadata={"help": "depths"},
    )
    dims: Optional[List[int]] = field(
        default_factory = lambda: [48, 96, 192, 384],
        metadata={"help": "dims"},
    )
    heads: Optional[List[int]] = field(
        default_factory = lambda: [1, 2, 4, 4],
        metadata={"help": "heads"},
    )
    num_slices_list: Optional[List[int]] = field(
        default_factory = lambda: [64, 32, 16, 8],
        metadata={"help": "num_slices_list"},
    )
    out_indices: Optional[List[int]] = field(
        default_factory = lambda: [0, 1, 2, 3],
        metadata={"help": "out_indices"},
    )
    
    