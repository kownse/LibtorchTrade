#include <torch/torch.h>

// AMD Ryzen is fastest for now
const torch::Device g_device = torch::kCPU;
// torch::cuda::is_available() ? torch::kCUDA : torch::kCPU
const auto g_datatype = at::kFloat;