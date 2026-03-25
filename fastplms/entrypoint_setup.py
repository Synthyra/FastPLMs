import torch
import torch._inductor.config as inductor_config
import torch._dynamo as dynamo

# Enable TensorFloat32 tensor cores for float32 matmul (Ampere+ GPUs)
# Provides significant speedup with minimal precision loss
torch.set_float32_matmul_precision('high')

# Enable TF32 for matrix multiplications and cuDNN operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN autotuner - finds fastest algorithms for your hardware
# Best when input sizes are consistent; may slow down first iterations
torch.backends.cudnn.benchmark = True

# Deterministic operations off for speed (set True if reproducibility needed)
torch.backends.cudnn.deterministic = False
inductor_config.max_autotune_gemm_backends = "ATEN,CUTLASS,FBGEMM"    

dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.recompile_limit = 16
