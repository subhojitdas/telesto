import os
import torch
from torch.utils.cpp_extension import load

# Optional: restrict archs if you want to avoid compiling for all visible devices
# os.environ["TORCH_CUDA_ARCH_LIST"] = "compute_86;sm_86"

# Build the extension from source files in the current directory.
# This will compile binding.cpp and kernel.cu and produce a Python module named "simple_add_ext".
ext = load(
    name="simple_add_ext",
    sources=["binding.cpp", "kernel.cu"],
    verbose=True,         # prints compiler output
    extra_cuda_cflags=['-lineinfo']  # optional debugging info
)

def test_add():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    device = torch.device("cuda")
    n = 1_000_000
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = torch.empty_like(a)

    # Call the extension
    ext.add_cuda(a, b, c)

    # (Optional) synchronize and validate
    torch.cuda.synchronize()
    expected = a + b
    if not torch.allclose(c, expected):
        print("Mismatch! max error:", (c - expected).abs().max().item())
    else:
        print("Success: c == a + b (first 6 elements):")
        print(a[:6].cpu().numpy())
        print(b[:6].cpu().numpy())
        print(c[:6].cpu().numpy())

if __name__ == "__main__":
    test_add()
