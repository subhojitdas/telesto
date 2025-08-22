#include <torch/extension.h>
#include <stdint.h>

// forward declaration of the launcher in kernel.cu
extern "C" void add_cuda_launcher(const float* a, const float* b, float* c, int64_t size);

void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.device().is_cuda(), "c must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "only float32 supported in this example");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "only float32 supported in this example");
    TORCH_CHECK(c.dtype() == torch::kFloat32, "only float32 supported in this example");
    TORCH_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "all tensors must have same number of elements");

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();
    int64_t size = a.numel();

    // call the launcher defined in kernel.cu
    add_cuda_launcher(a_ptr, b_ptr, c_ptr, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Elementwise add (CUDA)");
}