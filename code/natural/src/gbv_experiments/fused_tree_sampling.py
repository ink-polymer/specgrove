"""Lazy CUDA extension for one-launch ancestral probability-tree sampling."""
from __future__ import annotations

from functools import lru_cache
import math

import torch


CPP_SOURCE = r"""
torch::Tensor fused_tree_sample_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_sample_parallel_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_sample_scan_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_sample_sparse_exit_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_sample_sparse_exit_batched_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_follow_cuda(
    torch::Tensor posterior_tokens,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    int64_t max_depth);

torch::Tensor fused_tree_sample_logits_scan_cuda(
    torch::Tensor logits,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth);

torch::Tensor fused_tree_sample_logits_scan_batched_cuda(
    torch::Tensor logits,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth);

torch::Tensor fused_tree_sample_logits_sparse_exit_batched_cuda(
    torch::Tensor logits,
    torch::Tensor log_normalizers,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth);

torch::Tensor fused_internal_tree_sample_scan_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_internal_tree_sample_scan_batched_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_internal_tree_sample_sparse_exit_batched_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>

namespace cg = cooperative_groups;

__global__ void fused_tree_follow_kernel(
    const int64_t* __restrict__ posterior_tokens,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    int64_t* __restrict__ output,
    int node_count,
    int max_depth) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int current_node = 0;
  int accepted_count = 0;
  int bonus = static_cast<int>(posterior_tokens[0]);
  for (int depth = 0; depth < max_depth; ++depth) {
    int child = -1;
    for (int edge = 0; edge < node_count - 1; ++edge) {
      if (edge_parents[edge] == current_node
          && edge_tokens[edge] == bonus) {
        child = edge + 1;
        break;
      }
    }
    if (child < 0) {
      break;
    }
    output[accepted_count++] = child;
    current_node = child;
    bonus = static_cast<int>(posterior_tokens[current_node]);
  }
  output[max_depth] = accepted_count;
  output[max_depth + 1] = bonus;
}

torch::Tensor fused_tree_follow_cuda(
    torch::Tensor posterior_tokens,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    int64_t max_depth) {
  TORCH_CHECK(posterior_tokens.is_cuda()
              && posterior_tokens.scalar_type() == torch::kLong
              && posterior_tokens.is_contiguous()
              && posterior_tokens.dim() == 1,
              "posterior tokens must be contiguous CUDA long");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous CUDA long");
  const int64_t node_count = posterior_tokens.numel();
  TORCH_CHECK(node_count > 0 && edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 0 && max_depth <= INT_MAX
              && node_count <= INT_MAX,
              "tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(posterior_tokens.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong)
          .device(posterior_tokens.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_tree_follow_kernel<<<1, 1, 0, stream>>>(
      posterior_tokens.data_ptr<int64_t>(),
      edge_parents.data_ptr<int64_t>(),
      edge_tokens.data_ptr<int64_t>(),
      output.data_ptr<int64_t>(),
      static_cast<int>(node_count),
      static_cast<int>(max_depth));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t>
__global__ void fused_tree_sample_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int node_count,
    int vocabulary,
    int max_depth) {
  extern __shared__ double chunk_sums[];
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + blockDim.x - 1) / blockDim.x;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int row = current_node;
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      local_sum += static_cast<double>(probabilities[row * vocabulary + token]);
    }
    chunk_sums[thread] = local_sum;
    __syncthreads();

    if (thread == 0) {
      double row_total = 0.0;
      for (int lane = 0; lane < blockDim.x; ++lane) {
        row_total += chunk_sums[lane];
      }
      double threshold = uniforms[depth] * row_total;
      double prefix = 0.0;
      int selected_lane = blockDim.x - 1;
      for (int lane = 0; lane < blockDim.x; ++lane) {
        const double next = prefix + chunk_sums[lane];
        if (threshold < next || lane == blockDim.x - 1) {
          selected_lane = lane;
          break;
        }
        prefix = next;
      }
      int selected_token = min(selected_lane * chunk, vocabulary - 1);
      const int selected_end = min(selected_token + chunk, vocabulary);
      for (int token = selected_token; token < selected_end; ++token) {
        prefix += static_cast<double>(probabilities[row * vocabulary + token]);
        selected_token = token;
        if (threshold < prefix) {
          break;
        }
      }

      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == row && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0 && depth < max_depth) {
        output[accepted_count] = child;
        ++accepted_count;
        current_node = child;
      } else {
        output[max_depth] = accepted_count;
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
    }
    __syncthreads();
  }
}

torch::Tensor fused_tree_sample_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda(), "probabilities must be CUDA");
  TORCH_CHECK(probabilities.is_contiguous(), "probabilities must be contiguous");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(), "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(edge_parents.is_contiguous() && edge_tokens.is_contiguous(), "tree edges must be contiguous");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  const int64_t node_count = probabilities.size(0);
  const int64_t vocabulary = probabilities.size(1);
  TORCH_CHECK(edge_parents.numel() == node_count - 1 && edge_tokens.numel() == node_count - 1,
              "tree edge count mismatch");
  TORCH_CHECK(node_count > 0 && vocabulary > 0, "empty probability tree");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX && max_depth <= INT_MAX,
              "tree dimensions exceed CUDA kernel limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(probabilities.device()));
  constexpr int threads = 1024;
  const size_t shared_bytes = threads * sizeof(double);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(probabilities.scalar_type(), "fused_tree_sample_cuda", [&] {
    fused_tree_sample_kernel<scalar_t><<<1, threads, shared_bytes, stream>>>(
        probabilities.data_ptr<scalar_t>(),
        edge_parents.data_ptr<int64_t>(),
        edge_tokens.data_ptr<int64_t>(),
        uniforms.data_ptr<double>(),
        output.data_ptr<int64_t>(),
        static_cast<int>(node_count),
        static_cast<int>(vocabulary),
        static_cast<int>(max_depth));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

// Sparse routing from unchanged Target logits.  Torch computes one FP64
// log-normalizer per row; the kernel evaluates only outgoing tree edges while
// the walk stays inside the tree, then scans the vocabulary exactly once for
// the complement draw at the exit row.
template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_tree_sample_logits_sparse_exit_batched_kernel(
    const scalar_t* __restrict__ logits,
    const double* __restrict__ log_normalizers,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    double inverse_temperature,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int accepted_count;
  __shared__ int child_count;
  __shared__ int route_child;
  __shared__ int selected_token;
  __shared__ int child_tokens[64];

  const int request = blockIdx.x;
  logits += static_cast<int64_t>(request) * node_count * vocabulary;
  log_normalizers += static_cast<int64_t>(request) * node_count;
  edge_parents += static_cast<int64_t>(request) * (node_count - 1);
  edge_tokens += static_cast<int64_t>(request) * (node_count - 1);
  uniforms += static_cast<int64_t>(request) * 2 * (max_depth + 1);
  output += static_cast<int64_t>(request) * (max_depth + 2);

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    const int row = current_node;
    if (thread == 0) {
      child_count = 0;
      route_child = -1;
      double cumulative = 0.0;
      const double route_uniform = uniforms[2 * depth];
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] != row) {
          continue;
        }
        const int token = static_cast<int>(edge_tokens[edge]);
        child_tokens[child_count++] = token;
        const double scaled = static_cast<double>(
            logits[row * vocabulary + token]) * inverse_temperature;
        cumulative += exp(scaled - log_normalizers[row]);
        if (route_child < 0 && route_uniform < cumulative) {
          route_child = edge + 1;
        }
      }
      if (route_child >= 0) {
        output[accepted_count++] = route_child;
        current_node = route_child;
      }
    }
    __syncthreads();

    if (route_child >= 0) {
      continue;
    }

    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      bool is_child = false;
      for (int slot = 0; slot < child_count; ++slot) {
        is_child = is_child || child_tokens[slot] == token;
      }
      if (!is_child) {
        const double scaled = static_cast<double>(
            logits[row * vocabulary + token]) * inverse_temperature;
        local_sum += exp(scaled - log_normalizers[row]);
      }
    }

    double exclusive_prefix = 0.0;
    double complement_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, complement_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[2 * depth + 1] * complement_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        bool is_child = false;
        for (int slot = 0; slot < child_count; ++slot) {
          is_child = is_child || child_tokens[slot] == token;
        }
        if (is_child) {
          continue;
        }
        const double scaled = static_cast<double>(
            logits[row * vocabulary + token]) * inverse_temperature;
        prefix += exp(scaled - log_normalizers[row]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        for (int token = vocabulary - 1; token >= 0; --token) {
          bool is_child = false;
          for (int slot = 0; slot < child_count; ++slot) {
            is_child = is_child || child_tokens[slot] == token;
          }
          if (!is_child) {
            selected_token = token;
            break;
          }
        }
      }
      output[max_depth] = accepted_count;
      output[max_depth + 1] = selected_token;
    }
    return;
  }
}

torch::Tensor fused_tree_sample_logits_sparse_exit_batched_cuda(
    torch::Tensor logits,
    torch::Tensor log_normalizers,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous()
              && logits.dim() == 3,
              "batched sparse logits must be a contiguous CUDA tensor");
  TORCH_CHECK(log_normalizers.is_cuda() && log_normalizers.is_contiguous()
              && log_normalizers.scalar_type() == torch::kFloat64
              && log_normalizers.dim() == 2,
              "log normalizers must be a contiguous CUDA float64 matrix");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous()
              && edge_parents.dim() == 2 && edge_tokens.dim() == 2,
              "batched sparse-logit edges must be CUDA long matrices");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.is_contiguous()
              && uniforms.scalar_type() == torch::kFloat64
              && uniforms.dim() == 2,
              "batched sparse-logit uniforms must be CUDA float64");
  TORCH_CHECK(std::isfinite(temperature) && temperature > 0.0,
              "temperature must be finite and positive");
  const int64_t batch = logits.size(0);
  const int64_t node_count = logits.size(1);
  const int64_t vocabulary = logits.size(2);
  TORCH_CHECK(batch > 0 && node_count > 0 && node_count <= 65
              && vocabulary > 0,
              "batched sparse-logit sampler supports at most 64 edges");
  TORCH_CHECK(log_normalizers.size(0) == batch
              && log_normalizers.size(1) == node_count
              && edge_parents.size(0) == batch
              && edge_tokens.size(0) == batch
              && edge_parents.size(1) == node_count - 1
              && edge_tokens.size(1) == node_count - 1,
              "batched sparse-logit dimensions mismatch");
  TORCH_CHECK(max_depth >= 0 && uniforms.size(0) == batch
              && uniforms.size(1) >= 2 * (max_depth + 1),
              "batched sparse-logit uniforms do not cover every depth");
  TORCH_CHECK(batch <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX,
              "batched sparse-logit dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(logits.device());
  auto output = torch::full(
      {batch, max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  constexpr int threads = 640;
  const double inverse_temperature = 1.0 / temperature;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(),
      "fused_tree_sample_logits_sparse_exit_batched_cuda", [&] {
        fused_tree_sample_logits_sparse_exit_batched_kernel<scalar_t, threads>
            <<<static_cast<int>(batch), threads, 0, stream>>>(
                logits.data_ptr<scalar_t>(),
                log_normalizers.data_ptr<double>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                inverse_temperature,
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_tree_sample_scan_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int row = current_node;
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      local_sum += static_cast<double>(
          probabilities[row * vocabulary + token]);
    }

    double exclusive_prefix = 0.0;
    double row_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, row_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        prefix += static_cast<double>(
            probabilities[row * vocabulary + token]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      // torch.rand is strictly below one.  This fallback only covers a final
      // rounding gap in a non-normalized but otherwise positive input row.
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0 && depth < max_depth) {
        output[accepted_count] = child;
        ++accepted_count;
        current_node = child;
      } else {
        output[max_depth] = accepted_count;
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
    }
    __syncthreads();
  }
}

torch::Tensor fused_tree_sample_scan_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda(), "probabilities must be CUDA");
  TORCH_CHECK(probabilities.is_contiguous(), "probabilities must be contiguous");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(),
              "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous");
  TORCH_CHECK(uniforms.is_cuda()
              && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  const int64_t node_count = probabilities.size(0);
  const int64_t vocabulary = probabilities.size(1);
  TORCH_CHECK(edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree edge count mismatch");
  TORCH_CHECK(node_count > 0 && vocabulary > 0,
              "empty probability tree");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX, "tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      probabilities.scalar_type(), "fused_tree_sample_scan_cuda", [&] {
        fused_tree_sample_scan_kernel<scalar_t, threads>
            <<<1, threads, 0, stream>>>(
                probabilities.data_ptr<scalar_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

// Exact ancestral traversal with a sparse routing phase.  At an internal
// node, only probabilities carried by outgoing tree edges are needed to decide
// whether the Target draw remains in the tree.  A full-vocabulary scan is
// performed exactly once, at the first exit row, to draw the correction token
// from the complement of that row's children.
template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_tree_sample_sparse_exit_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int accepted_count;
  __shared__ int child_count;
  __shared__ int route_child;
  __shared__ int selected_token;
  // Formal DDTree uses B=45.  The generic guard below permits up to 64 edges
  // while keeping the routing table resident in shared memory.
  __shared__ int child_tokens[64];
  __shared__ int child_nodes[64];

  const int request = blockIdx.x;
  probabilities += static_cast<int64_t>(request) * node_count * vocabulary;
  edge_parents += static_cast<int64_t>(request) * (node_count - 1);
  edge_tokens += static_cast<int64_t>(request) * (node_count - 1);
  uniforms += static_cast<int64_t>(request) * 2 * (max_depth + 1);
  output += static_cast<int64_t>(request) * (max_depth + 2);

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    const int row = current_node;
    if (thread == 0) {
      child_count = 0;
      route_child = -1;
      double cumulative = 0.0;
      const double route_uniform = uniforms[2 * depth];
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] != row) {
          continue;
        }
        const int slot = child_count++;
        const int token = static_cast<int>(edge_tokens[edge]);
        child_tokens[slot] = token;
        child_nodes[slot] = edge + 1;
        cumulative += static_cast<double>(
            probabilities[row * vocabulary + token]);
        if (route_child < 0 && route_uniform < cumulative) {
          route_child = edge + 1;
        }
      }
      if (route_child >= 0) {
        output[accepted_count++] = route_child;
        current_node = route_child;
      }
    }
    __syncthreads();

    if (route_child >= 0) {
      continue;
    }

    // The route left the finite tree.  Sample the correction token from the
    // exact Target row with the current node's child tokens removed.
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      bool is_child = false;
      for (int slot = 0; slot < child_count; ++slot) {
        is_child = is_child || child_tokens[slot] == token;
      }
      if (!is_child) {
        local_sum += static_cast<double>(
            probabilities[row * vocabulary + token]);
      }
    }

    double exclusive_prefix = 0.0;
    double complement_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, complement_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[2 * depth + 1] * complement_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        bool is_child = false;
        for (int slot = 0; slot < child_count; ++slot) {
          is_child = is_child || child_tokens[slot] == token;
        }
        if (is_child) {
          continue;
        }
        prefix += static_cast<double>(
            probabilities[row * vocabulary + token]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      // A softmax row has positive complement mass.  The fallback only covers
      // a final floating-point boundary and deliberately avoids child tokens.
      if (selected_token < 0) {
        for (int token = vocabulary - 1; token >= 0; --token) {
          bool is_child = false;
          for (int slot = 0; slot < child_count; ++slot) {
            is_child = is_child || child_tokens[slot] == token;
          }
          if (!is_child) {
            selected_token = token;
            break;
          }
        }
      }
      output[max_depth] = accepted_count;
      output[max_depth + 1] = selected_token;
    }
    return;
  }
}

torch::Tensor fused_tree_sample_sparse_exit_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda() && probabilities.is_contiguous()
              && probabilities.dim() == 2,
              "probabilities must be a contiguous CUDA matrix");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous CUDA long");
  TORCH_CHECK(uniforms.is_cuda()
              && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  const int64_t node_count = probabilities.size(0);
  const int64_t vocabulary = probabilities.size(1);
  TORCH_CHECK(node_count > 0 && node_count <= 65 && vocabulary > 0,
              "sparse-exit sampler supports one root plus at most 64 edges");
  TORCH_CHECK(edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree edge count mismatch");
  TORCH_CHECK(max_depth >= 0
              && uniforms.numel() >= 2 * (max_depth + 1),
              "not enough uniforms for sparse-exit traversal");
  TORCH_CHECK(vocabulary <= INT_MAX && max_depth <= INT_MAX,
              "tree dimensions exceed CUDA kernel limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong)
          .device(probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      probabilities.scalar_type(), "fused_tree_sample_sparse_exit_cuda", [&] {
        fused_tree_sample_sparse_exit_kernel<scalar_t, threads>
            <<<1, threads, 0, stream>>>(
                probabilities.data_ptr<scalar_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fused_tree_sample_sparse_exit_batched_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda() && probabilities.is_contiguous()
              && probabilities.dim() == 3,
              "batched sparse probabilities must be a contiguous CUDA tensor");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous()
              && edge_parents.dim() == 2 && edge_tokens.dim() == 2,
              "batched sparse tree edges must be CUDA long matrices");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.is_contiguous()
              && uniforms.scalar_type() == torch::kFloat64
              && uniforms.dim() == 2,
              "batched sparse uniforms must be a CUDA float64 matrix");
  const int64_t batch = probabilities.size(0);
  const int64_t node_count = probabilities.size(1);
  const int64_t vocabulary = probabilities.size(2);
  TORCH_CHECK(batch > 0 && node_count > 0 && node_count <= 65
              && vocabulary > 0,
              "batched sparse sampler supports at most 64 tree edges");
  TORCH_CHECK(edge_parents.size(0) == batch
              && edge_tokens.size(0) == batch
              && edge_parents.size(1) == node_count - 1
              && edge_tokens.size(1) == node_count - 1,
              "batched sparse tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 0 && uniforms.size(0) == batch
              && uniforms.size(1) >= 2 * (max_depth + 1),
              "batched sparse uniforms do not cover every depth");
  TORCH_CHECK(batch <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX,
              "batched sparse tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto output = torch::full(
      {batch, max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      probabilities.scalar_type(),
      "fused_tree_sample_sparse_exit_batched_cuda", [&] {
        fused_tree_sample_sparse_exit_kernel<scalar_t, threads>
            <<<static_cast<int>(batch), threads, 0, stream>>>(
                probabilities.data_ptr<scalar_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_tree_sample_logits_scan_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    double inverse_temperature,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_THREADS>;
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  union Scratch {
    typename BlockReduce::TempStorage reduce;
    typename BlockScan::TempStorage scan;
  };
  __shared__ Scratch scratch;
  __shared__ double row_max_shared;
  __shared__ double row_total_shared;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int row = current_node;
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_max = -DBL_MAX;
    for (int token = begin; token < end; ++token) {
      local_max = max(
          local_max,
          static_cast<double>(logits[row * vocabulary + token])
              * inverse_temperature);
    }
    const double reduced_max = BlockReduce(scratch.reduce).Reduce(
        local_max, cub::Max());
    if (thread == 0) {
      row_max_shared = reduced_max;
    }
    __syncthreads();

    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      const double scaled =
          static_cast<double>(logits[row * vocabulary + token])
              * inverse_temperature;
      local_sum += exp(scaled - row_max_shared);
    }
    double exclusive_prefix = 0.0;
    double reduced_total = 0.0;
    BlockScan(scratch.scan).ExclusiveSum(
        local_sum, exclusive_prefix, reduced_total);
    if (thread == 0) {
      row_total_shared = reduced_total;
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total_shared;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        const double scaled =
            static_cast<double>(logits[row * vocabulary + token])
                * inverse_temperature;
        prefix += exp(scaled - row_max_shared);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0 && depth < max_depth) {
        output[accepted_count] = child;
        ++accepted_count;
        current_node = child;
      } else {
        output[max_depth] = accepted_count;
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
    }
    __syncthreads();
  }
}

torch::Tensor fused_tree_sample_logits_scan_cuda(
    torch::Tensor logits,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
              "logits must be contiguous CUDA");
  TORCH_CHECK(logits.dim() == 2, "logits must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(),
              "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous");
  TORCH_CHECK(uniforms.is_cuda()
              && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  TORCH_CHECK(std::isfinite(temperature) && temperature > 0.0,
              "temperature must be finite and positive");
  const int64_t node_count = logits.size(0);
  const int64_t vocabulary = logits.size(1);
  TORCH_CHECK(node_count > 0 && vocabulary > 0,
              "empty logit tree");
  TORCH_CHECK(edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree edge count mismatch");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX, "tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(logits.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  constexpr int threads = 640;
  const double inverse_temperature = 1.0 / temperature;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(),
      "fused_tree_sample_logits_scan_cuda", [&] {
        fused_tree_sample_logits_scan_kernel<scalar_t, threads>
            <<<1, threads, 0, stream>>>(
                logits.data_ptr<scalar_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                inverse_temperature,
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_tree_sample_logits_scan_batched_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    double inverse_temperature,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_THREADS>;
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  union Scratch {
    typename BlockReduce::TempStorage reduce;
    typename BlockScan::TempStorage scan;
  };
  __shared__ Scratch scratch;
  __shared__ double row_max_shared;
  __shared__ double row_total_shared;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int request = blockIdx.x;
  logits += static_cast<int64_t>(request) * node_count * vocabulary;
  edge_parents += static_cast<int64_t>(request) * (node_count - 1);
  edge_tokens += static_cast<int64_t>(request) * (node_count - 1);
  uniforms += static_cast<int64_t>(request) * (max_depth + 1);
  output += static_cast<int64_t>(request) * (max_depth + 2);

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int row = current_node;
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_max = -DBL_MAX;
    for (int token = begin; token < end; ++token) {
      local_max = max(
          local_max,
          static_cast<double>(logits[row * vocabulary + token])
              * inverse_temperature);
    }
    const double reduced_max = BlockReduce(scratch.reduce).Reduce(
        local_max, cub::Max());
    if (thread == 0) {
      row_max_shared = reduced_max;
    }
    __syncthreads();

    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      const double scaled =
          static_cast<double>(logits[row * vocabulary + token])
              * inverse_temperature;
      local_sum += exp(scaled - row_max_shared);
    }
    double exclusive_prefix = 0.0;
    double reduced_total = 0.0;
    BlockScan(scratch.scan).ExclusiveSum(
        local_sum, exclusive_prefix, reduced_total);
    if (thread == 0) {
      row_total_shared = reduced_total;
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total_shared;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        const double scaled =
            static_cast<double>(logits[row * vocabulary + token])
                * inverse_temperature;
        prefix += exp(scaled - row_max_shared);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0 && depth < max_depth) {
        output[accepted_count] = child;
        ++accepted_count;
        current_node = child;
      } else {
        output[max_depth] = accepted_count;
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
    }
    __syncthreads();
  }
}

torch::Tensor fused_tree_sample_logits_scan_batched_cuda(
    torch::Tensor logits,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous()
              && logits.dim() == 3,
              "batched logits must be a contiguous CUDA tensor");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous()
              && edge_parents.dim() == 2 && edge_tokens.dim() == 2,
              "batched tree edges must be contiguous CUDA long matrices");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.is_contiguous()
              && uniforms.scalar_type() == torch::kFloat64
              && uniforms.dim() == 2,
              "batched uniforms must be a contiguous CUDA float64 matrix");
  TORCH_CHECK(std::isfinite(temperature) && temperature > 0.0,
              "temperature must be finite and positive");
  const int64_t batch = logits.size(0);
  const int64_t node_count = logits.size(1);
  const int64_t vocabulary = logits.size(2);
  TORCH_CHECK(batch > 0 && node_count > 0 && vocabulary > 0,
              "empty batched logit tree");
  TORCH_CHECK(edge_parents.size(0) == batch
              && edge_tokens.size(0) == batch
              && edge_parents.size(1) == node_count - 1
              && edge_tokens.size(1) == node_count - 1,
              "batched tree edge dimensions mismatch");
  TORCH_CHECK(max_depth >= 0 && uniforms.size(0) == batch
              && uniforms.size(1) >= max_depth + 1,
              "batched uniforms do not cover every tree depth");
  TORCH_CHECK(batch <= INT_MAX && node_count <= INT_MAX
              && vocabulary <= INT_MAX && max_depth <= INT_MAX,
              "batched tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(logits.device());
  auto output = torch::full(
      {batch, max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  constexpr int threads = 640;
  const double inverse_temperature = 1.0 / temperature;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(),
      "fused_tree_sample_logits_scan_batched_cuda", [&] {
        fused_tree_sample_logits_scan_batched_kernel<scalar_t, threads>
            <<<static_cast<int>(batch), threads, 0, stream>>>(
                logits.data_ptr<scalar_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                inverse_temperature,
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_internal_tree_sample_scan_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ internal_rows,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int original_row = current_node;
    const int compact_row = static_cast<int>(internal_rows[original_row]);
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      local_sum += static_cast<double>(
          probabilities[compact_row * vocabulary + token]);
    }

    double exclusive_prefix = 0.0;
    double row_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, row_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        prefix += static_cast<double>(
            probabilities[compact_row * vocabulary + token]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == original_row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0) {
        output[accepted_count] = child;
        ++accepted_count;
        if (internal_rows[child] >= 0 && depth < max_depth) {
          current_node = child;
        } else {
          // The selected child is a leaf.  Its probability row was deliberately
          // not materialized; the caller samples that one row on demand.
          output[max_depth + 2] = child;
          finished = 1;
        }
      } else {
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
      output[max_depth] = accepted_count;
    }
    __syncthreads();
  }
}

torch::Tensor fused_internal_tree_sample_scan_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(internal_probabilities.is_cuda()
              && internal_probabilities.is_contiguous(),
              "internal probabilities must be contiguous CUDA");
  TORCH_CHECK(internal_probabilities.dim() == 2
              && internal_probabilities.size(0) > 0
              && internal_probabilities.size(1) > 0,
              "internal probabilities must be a nonempty matrix");
  TORCH_CHECK(internal_rows.is_cuda()
              && internal_rows.scalar_type() == torch::kLong
              && internal_rows.is_contiguous(),
              "internal row map must be contiguous CUDA long");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous CUDA long");
  TORCH_CHECK(uniforms.is_cuda()
              && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  const int64_t node_count = internal_rows.numel();
  const int64_t vocabulary = internal_probabilities.size(1);
  TORCH_CHECK(node_count > 1 && edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 1 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX, "tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(internal_probabilities.device());
  auto output = torch::full(
      {max_depth + 3}, -1,
      torch::TensorOptions().dtype(torch::kLong)
          .device(internal_probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      internal_probabilities.scalar_type(),
      "fused_internal_tree_sample_scan_cuda", [&] {
        fused_internal_tree_sample_scan_kernel<scalar_t, threads>
            <<<1, threads, 0, stream>>>(
                internal_probabilities.data_ptr<scalar_t>(),
                internal_rows.data_ptr<int64_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_internal_tree_sample_scan_batched_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ internal_rows,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int internal_row_capacity,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int request = blockIdx.x;
  probabilities += static_cast<int64_t>(request)
      * internal_row_capacity * vocabulary;
  internal_rows += static_cast<int64_t>(request) * node_count;
  edge_parents += static_cast<int64_t>(request) * (node_count - 1);
  edge_tokens += static_cast<int64_t>(request) * (node_count - 1);
  uniforms += static_cast<int64_t>(request) * (max_depth + 1);
  output += static_cast<int64_t>(request) * (max_depth + 3);

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int original_row = current_node;
    const int compact_row = static_cast<int>(internal_rows[original_row]);
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      local_sum += static_cast<double>(
          probabilities[compact_row * vocabulary + token]);
    }

    double exclusive_prefix = 0.0;
    double row_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, row_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        prefix += static_cast<double>(
            probabilities[compact_row * vocabulary + token]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == original_row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0) {
        output[accepted_count] = child;
        ++accepted_count;
        if (internal_rows[child] >= 0 && depth < max_depth) {
          current_node = child;
        } else {
          output[max_depth + 2] = child;
          finished = 1;
        }
      } else {
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
      output[max_depth] = accepted_count;
    }
    __syncthreads();
  }
}

torch::Tensor fused_internal_tree_sample_scan_batched_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(internal_probabilities.is_cuda()
              && internal_probabilities.is_contiguous()
              && internal_probabilities.dim() == 3,
              "batched internal probabilities must be contiguous CUDA");
  TORCH_CHECK(internal_rows.is_cuda()
              && internal_rows.scalar_type() == torch::kLong
              && internal_rows.is_contiguous() && internal_rows.dim() == 2,
              "batched internal maps must be contiguous CUDA long matrices");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous()
              && edge_parents.dim() == 2 && edge_tokens.dim() == 2,
              "batched tree edges must be contiguous CUDA long matrices");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.is_contiguous()
              && uniforms.scalar_type() == torch::kFloat64
              && uniforms.dim() == 2,
              "batched uniforms must be contiguous CUDA float64 matrices");
  const int64_t batch = internal_probabilities.size(0);
  const int64_t internal_capacity = internal_probabilities.size(1);
  const int64_t vocabulary = internal_probabilities.size(2);
  const int64_t node_count = internal_rows.size(1);
  TORCH_CHECK(batch > 0 && internal_capacity > 0 && vocabulary > 0
              && node_count > 1,
              "empty batched internal probability tree");
  TORCH_CHECK(internal_rows.size(0) == batch
              && edge_parents.size(0) == batch
              && edge_tokens.size(0) == batch
              && edge_parents.size(1) == node_count - 1
              && edge_tokens.size(1) == node_count - 1,
              "batched internal tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 1 && uniforms.size(0) == batch
              && uniforms.size(1) >= max_depth + 1,
              "batched internal uniforms do not cover every depth");
  TORCH_CHECK(batch <= INT_MAX && internal_capacity <= INT_MAX
              && node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX,
              "batched internal tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(internal_probabilities.device());
  auto output = torch::full(
      {batch, max_depth + 3}, -1,
      torch::TensorOptions().dtype(torch::kLong)
          .device(internal_probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      internal_probabilities.scalar_type(),
      "fused_internal_tree_sample_scan_batched_cuda", [&] {
        fused_internal_tree_sample_scan_batched_kernel<scalar_t, threads>
            <<<static_cast<int>(batch), threads, 0, stream>>>(
                internal_probabilities.data_ptr<scalar_t>(),
                internal_rows.data_ptr<int64_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(internal_capacity),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

// Batched counterpart of sparse-exit verification for compact internal rows.
// Every accepted internal node touches only its outgoing child probabilities.
// The vocabulary is scanned once, and only once, when a request exits from an
// internal row.  Reaching a leaf is reported to Python so all leaf vocabulary
// projections can be coalesced into one GEMM.
template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_internal_tree_sample_sparse_exit_batched_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ internal_rows,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int internal_row_capacity,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int accepted_count;
  __shared__ int child_count;
  __shared__ int route_child;
  __shared__ int finished;
  __shared__ int selected_token;
  __shared__ int child_tokens[64];

  const int request = blockIdx.x;
  probabilities += static_cast<int64_t>(request)
      * internal_row_capacity * vocabulary;
  internal_rows += static_cast<int64_t>(request) * node_count;
  edge_parents += static_cast<int64_t>(request) * (node_count - 1);
  edge_tokens += static_cast<int64_t>(request) * (node_count - 1);
  uniforms += static_cast<int64_t>(request) * 2 * (max_depth + 1);
  output += static_cast<int64_t>(request) * (max_depth + 3);

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    accepted_count = 0;
    finished = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    const int original_row = current_node;
    const int compact_row = static_cast<int>(internal_rows[original_row]);
    if (thread == 0) {
      child_count = 0;
      route_child = -1;
      double cumulative = 0.0;
      const double route_uniform = uniforms[2 * depth];
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] != original_row) {
          continue;
        }
        const int token = static_cast<int>(edge_tokens[edge]);
        child_tokens[child_count++] = token;
        cumulative += static_cast<double>(
            probabilities[compact_row * vocabulary + token]);
        if (route_child < 0 && route_uniform < cumulative) {
          route_child = edge + 1;
        }
      }
      if (route_child >= 0) {
        output[accepted_count++] = route_child;
        output[max_depth] = accepted_count;
        if (internal_rows[route_child] >= 0 && depth < max_depth) {
          current_node = route_child;
        } else {
          output[max_depth + 2] = route_child;
          finished = 1;
        }
      }
    }
    __syncthreads();

    if (finished) {
      return;
    }
    if (route_child >= 0) {
      continue;
    }

    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      bool is_child = false;
      for (int slot = 0; slot < child_count; ++slot) {
        is_child = is_child || child_tokens[slot] == token;
      }
      if (!is_child) {
        local_sum += static_cast<double>(
            probabilities[compact_row * vocabulary + token]);
      }
    }

    double exclusive_prefix = 0.0;
    double complement_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, complement_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[2 * depth + 1] * complement_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        bool is_child = false;
        for (int slot = 0; slot < child_count; ++slot) {
          is_child = is_child || child_tokens[slot] == token;
        }
        if (is_child) {
          continue;
        }
        prefix += static_cast<double>(
            probabilities[compact_row * vocabulary + token]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        for (int token = vocabulary - 1; token >= 0; --token) {
          bool is_child = false;
          for (int slot = 0; slot < child_count; ++slot) {
            is_child = is_child || child_tokens[slot] == token;
          }
          if (!is_child) {
            selected_token = token;
            break;
          }
        }
      }
      output[max_depth] = accepted_count;
      output[max_depth + 1] = selected_token;
    }
    return;
  }
}

torch::Tensor fused_internal_tree_sample_sparse_exit_batched_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(internal_probabilities.is_cuda()
              && internal_probabilities.is_contiguous()
              && internal_probabilities.dim() == 3,
              "batched sparse internal probabilities must be contiguous CUDA");
  TORCH_CHECK(internal_rows.is_cuda()
              && internal_rows.scalar_type() == torch::kLong
              && internal_rows.is_contiguous() && internal_rows.dim() == 2,
              "batched sparse internal maps must be CUDA long matrices");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous()
              && edge_parents.dim() == 2 && edge_tokens.dim() == 2,
              "batched sparse tree edges must be CUDA long matrices");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.is_contiguous()
              && uniforms.scalar_type() == torch::kFloat64
              && uniforms.dim() == 2,
              "batched sparse uniforms must be CUDA float64 matrices");
  const int64_t batch = internal_probabilities.size(0);
  const int64_t internal_capacity = internal_probabilities.size(1);
  const int64_t vocabulary = internal_probabilities.size(2);
  const int64_t node_count = internal_rows.size(1);
  TORCH_CHECK(batch > 0 && internal_capacity > 0 && vocabulary > 0
              && node_count > 1 && node_count <= 65,
              "batched sparse internal tree supports at most 64 edges");
  TORCH_CHECK(internal_rows.size(0) == batch
              && edge_parents.size(0) == batch
              && edge_tokens.size(0) == batch
              && edge_parents.size(1) == node_count - 1
              && edge_tokens.size(1) == node_count - 1,
              "batched sparse internal tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 1 && uniforms.size(0) == batch
              && uniforms.size(1) >= 2 * (max_depth + 1),
              "batched sparse uniforms do not cover every depth");
  TORCH_CHECK(batch <= INT_MAX && internal_capacity <= INT_MAX
              && node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX,
              "batched sparse internal dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(internal_probabilities.device());
  auto output = torch::full(
      {batch, max_depth + 3}, -1,
      torch::TensorOptions().dtype(torch::kLong)
          .device(internal_probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      internal_probabilities.scalar_type(),
      "fused_internal_tree_sample_sparse_exit_batched_cuda", [&] {
        fused_internal_tree_sample_sparse_exit_batched_kernel<scalar_t, threads>
            <<<static_cast<int>(batch), threads, 0, stream>>>(
                internal_probabilities.data_ptr<scalar_t>(),
                internal_rows.data_ptr<int64_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(internal_capacity),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t>
__global__ void fused_tree_sample_cooperative_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ state,
    double* __restrict__ workspace,
    int node_count,
    int vocabulary,
    int max_depth) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ double lane_sums[];
  const int block_chunk = (vocabulary + gridDim.x - 1) / gridDim.x;

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (state[max_depth + 3]) {
      break;
    }
    const int row = static_cast<int>(state[max_depth + 2]);
    const int begin = blockIdx.x * block_chunk;
    const int end = min(begin + block_chunk, vocabulary);
    double local = 0.0;
    for (int token = begin + threadIdx.x; token < end; token += blockDim.x) {
      local += static_cast<double>(probabilities[row * vocabulary + token]);
    }
    lane_sums[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        lane_sums[threadIdx.x] += lane_sums[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      workspace[blockIdx.x] = lane_sums[0];
    }
    grid.sync();

    // One thread chooses the vocabulary partition.  Its residual threshold is
    // then consumed by that partition's block, so no second kernel launch or
    // host decision is needed between tree depths.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      double total = 0.0;
      for (int block = 0; block < gridDim.x; ++block) {
        total += workspace[block];
      }
      const double threshold = uniforms[depth] * total;
      double prefix = 0.0;
      int selected_block = gridDim.x - 1;
      for (int block = 0; block < gridDim.x; ++block) {
        const double next = prefix + workspace[block];
        if (threshold < next || block == gridDim.x - 1) {
          selected_block = block;
          workspace[gridDim.x] = threshold - prefix;
          break;
        }
        prefix = next;
      }
      state[max_depth + 4] = selected_block;
    }
    grid.sync();

    const int selected_block = static_cast<int>(state[max_depth + 4]);
    if (blockIdx.x == selected_block) {
      const int selected_begin = selected_block * block_chunk;
      const int selected_end = min(selected_begin + block_chunk, vocabulary);
      const int lane_chunk =
          (selected_end - selected_begin + blockDim.x - 1) / blockDim.x;
      const int lane_begin = selected_begin + threadIdx.x * lane_chunk;
      const int lane_end = min(lane_begin + lane_chunk, selected_end);
      double selected_local = 0.0;
      for (int token = lane_begin; token < lane_end; ++token) {
        selected_local +=
            static_cast<double>(probabilities[row * vocabulary + token]);
      }
      lane_sums[threadIdx.x] = selected_local;
      __syncthreads();

      if (threadIdx.x == 0) {
        const double local_threshold = workspace[gridDim.x];
        double prefix = 0.0;
        int selected_lane = blockDim.x - 1;
        for (int lane = 0; lane < blockDim.x; ++lane) {
          const double next = prefix + lane_sums[lane];
          if (local_threshold < next || lane == blockDim.x - 1) {
            selected_lane = lane;
            break;
          }
          prefix = next;
        }
        int selected_token = min(
            selected_begin + selected_lane * lane_chunk, vocabulary - 1);
        const int token_end = min(selected_token + lane_chunk, selected_end);
        for (int token = selected_token; token < token_end; ++token) {
          prefix += static_cast<double>(probabilities[row * vocabulary + token]);
          selected_token = token;
          if (local_threshold < prefix) {
            break;
          }
        }

        int child = -1;
        for (int edge = 0; edge < node_count - 1; ++edge) {
          if (edge_parents[edge] == row && edge_tokens[edge] == selected_token) {
            child = edge + 1;
            break;
          }
        }
        const int accepted = static_cast<int>(state[max_depth]);
        if (child >= 0 && depth < max_depth) {
          state[accepted] = child;
          state[max_depth] = accepted + 1;
          state[max_depth + 2] = child;
        } else {
          state[max_depth + 1] = selected_token;
          state[max_depth + 3] = 1;
        }
      }
    }
    grid.sync();
  }
}

torch::Tensor fused_tree_sample_parallel_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda() && probabilities.is_contiguous(),
              "probabilities must be contiguous CUDA");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(), "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  const int64_t node_count = probabilities.size(0);
  const int64_t vocabulary = probabilities.size(1);
  TORCH_CHECK(node_count > 0 && vocabulary > 0 && edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1, "tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX && max_depth <= INT_MAX,
              "tree dimensions exceed CUDA kernel limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto state = torch::zeros(
      {max_depth + 5},
      torch::TensorOptions().dtype(torch::kLong).device(probabilities.device()));
  constexpr int requested_blocks = 64;
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(probabilities.scalar_type(), "fused_tree_sample_parallel_cuda", [&] {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int cooperative = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &cooperative, cudaDevAttrCooperativeLaunch, device));
    TORCH_CHECK(cooperative, "GPU does not support cooperative launch");
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, fused_tree_sample_cooperative_kernel<scalar_t>,
        threads, threads * sizeof(double)));
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const int blocks = std::min(
        requested_blocks, blocks_per_sm * properties.multiProcessorCount);
    TORCH_CHECK(blocks > 0, "No cooperative kernel occupancy available");
    auto workspace = torch::empty(
        {blocks + 1},
        torch::TensorOptions().dtype(torch::kFloat64).device(probabilities.device()));
    const scalar_t* probability_ptr = probabilities.data_ptr<scalar_t>();
    const int64_t* parent_ptr = edge_parents.data_ptr<int64_t>();
    const int64_t* token_ptr = edge_tokens.data_ptr<int64_t>();
    const double* uniform_ptr = uniforms.data_ptr<double>();
    int64_t* state_ptr = state.data_ptr<int64_t>();
    double* workspace_ptr = workspace.data_ptr<double>();
    int nodes = static_cast<int>(node_count);
    int vocab = static_cast<int>(vocabulary);
    int depth = static_cast<int>(max_depth);
    void* arguments[] = {
        &probability_ptr, &parent_ptr, &token_ptr, &uniform_ptr,
        &state_ptr, &workspace_ptr, &nodes, &vocab, &depth};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(fused_tree_sample_cooperative_kernel<scalar_t>),
        dim3(blocks), dim3(threads), arguments,
        threads * sizeof(double), stream));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return state;
}
"""


@lru_cache(maxsize=1)
def load_fused_tree_sampler():
    if not torch.cuda.is_available():
        raise RuntimeError("The fused tree sampler requires CUDA")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="gbv_fused_tree_sampler_v20",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        functions=["fused_tree_sample_cuda", "fused_tree_sample_parallel_cuda",
                   "fused_tree_sample_scan_cuda",
                   "fused_tree_sample_sparse_exit_cuda",
                   "fused_tree_sample_sparse_exit_batched_cuda",
                   "fused_tree_follow_cuda",
                   "fused_tree_sample_logits_scan_cuda",
                   "fused_tree_sample_logits_scan_batched_cuda",
                   "fused_tree_sample_logits_sparse_exit_batched_cuda",
                   "fused_internal_tree_sample_scan_cuda",
                   "fused_internal_tree_sample_scan_batched_cuda",
                   "fused_internal_tree_sample_sparse_exit_batched_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def _topology(parents, tokens):
    parents = list(parents)
    tokens = list(tokens)
    node_count = len(parents)
    if (node_count < 1 or len(tokens) != node_count - 1
            or parents[0] != -1
            or any(parent < 0 or parent >= node
                   for node, parent in enumerate(parents[1:], 1))
            or len(set(zip(parents[1:], tokens))) != len(tokens)):
        raise ValueError("Invalid probability-tree topology")
    depths = [0] * node_count
    for node in range(1, node_count):
        depths[node] = depths[parents[node]] + 1
    return parents, tokens, max(depths)


def tree_verify_ancestral_fused(parents, tokens, all_p, generator=None,
                                validate: bool = True):
    """Sample only the visited DDTree rows inside one persistent CUDA block."""
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[1] < 1 or not all_p.is_floating_point()):
        raise ValueError("Fused DDTree probability tensor mismatch")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for fused DDTree")

    device = all_p.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator
    )
    packed = load_fused_tree_sampler().fused_tree_sample_cuda(
        all_p.contiguous(), edge_parents, edge_tokens, uniforms, max_depth
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Fused tree sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    output_tokens = [tokens[node - 1] for node in nodes]
    return nodes, output_tokens, bonus


def tree_verify_ancestral_fused_parallel(parents, tokens, all_p, generator=None,
                                         validate: bool = True):
    """Traverse on device with multi-SM row reductions and no host decisions."""
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[1] < 1 or not all_p.is_floating_point()):
        raise ValueError("Parallel fused DDTree probability tensor mismatch")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for parallel fused DDTree")

    device = all_p.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator
    )
    packed = load_fused_tree_sampler().fused_tree_sample_parallel_cuda(
        all_p.contiguous(), edge_parents, edge_tokens, uniforms, max_depth
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Parallel fused tree sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus


def tree_verify_ancestral_fused_scan(parents, tokens, all_p, generator=None,
                                      validate: bool = True):
    """Traverse DDTree with one persistent block and a parallel CUB row scan."""
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[1] < 1 or not all_p.is_floating_point()):
        raise ValueError("Scan-fused DDTree probability tensor mismatch")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for scan-fused DDTree")

    device = all_p.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator
    )
    packed = load_fused_tree_sampler().fused_tree_sample_scan_cuda(
        all_p.contiguous(), edge_parents, edge_tokens, uniforms, max_depth
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Scan-fused tree sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus


def tree_verify_ancestral_sparse_exit_fused_scan(
        parents, tokens, all_p, generator=None, validate: bool = True):
    """Traverse sparse child events and scan the vocabulary only at exit.

    For every internal row this first draws from ``children U {exit}``.  On an
    exit it draws once more from the Target distribution conditioned on not
    selecting a child.  Hence each child edge and each correction token has
    exactly its Target probability, while accepted internal rows avoid a full
    vocabulary scan.
    """
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2
            or all_p.shape[0] != node_count or all_p.shape[1] < 1
            or not all_p.is_floating_point() or not all_p.is_contiguous()):
        raise ValueError("Sparse-exit Target probability tensor mismatch")
    if node_count > 65:
        raise ValueError("Sparse-exit verifier supports at most 64 tree edges")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & torch.isclose(
                     all_p.sum(-1), all_p.new_ones(node_count),
                     rtol=1e-10, atol=1e-12,
                 ).all())
        if not bool(valid):
            raise FloatingPointError(
                "Sparse-exit verifier requires normalized Target probabilities"
            )

    device = all_p.device
    metadata = torch.tensor(
        [parents[1:], tokens], dtype=torch.long, device=device,
    )
    uniforms = torch.rand(
        2 * (max_depth + 1), dtype=torch.float64,
        device=device, generator=generator,
    )
    packed = load_fused_tree_sampler().fused_tree_sample_sparse_exit_cuda(
        all_p, metadata[0], metadata[1], uniforms, max_depth,
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Sparse-exit sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus


def tree_verify_ancestral_sparse_exit_fused_scan_batched(
        parents_batch, tokens_batch, probability_batch, generators,
        validate: bool = True):
    """Run equally shaped requests in one grid and synchronize once per wave.

    Request RNG streams and tree metadata remain independent.  Equally shaped
    trees use one multi-block launch and one control transfer.  Ragged batches
    retain the older queued-kernel fallback.
    """
    count = len(parents_batch)
    if (count < 1 or len(tokens_batch) != count
            or len(probability_batch) != count or len(generators) != count):
        raise ValueError("Sparse-exit batch inputs have different lengths")

    topologies = []
    probability_rows = (
        list(probability_batch.unbind(0))
        if isinstance(probability_batch, torch.Tensor)
        else list(probability_batch)
    )
    for parents, tokens, all_p, generator in zip(
            parents_batch, tokens_batch, probability_rows, generators):
        parents, tokens, max_depth = _topology(parents, tokens)
        node_count = len(parents)
        if (not all_p.is_cuda or all_p.ndim != 2
                or all_p.shape[0] != node_count or all_p.shape[1] < 1
                or not all_p.is_floating_point() or not all_p.is_contiguous()):
            raise ValueError("Sparse-exit Target probability tensor mismatch")
        if node_count > 65:
            raise ValueError("Sparse-exit verifier supports at most 64 tree edges")
        if any(token < 0 or token >= all_p.shape[1] for token in tokens):
            raise ValueError("Tree token is outside the Target vocabulary")
        if validate:
            valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                     & torch.isclose(
                         all_p.sum(-1), all_p.new_ones(node_count),
                         rtol=1e-10, atol=1e-12,
                     ).all())
            if not bool(valid):
                raise FloatingPointError(
                    "Sparse-exit verifier requires normalized Target probabilities"
                )
        topologies.append((parents, tokens, max_depth, all_p.shape[1]))

    extension = load_fused_tree_sampler()
    common_shape = (
        len({len(parents) for parents, _, _, _ in topologies}) == 1
        and len({depth for _, _, depth, _ in topologies}) == 1
        and len({vocab for _, _, _, vocab in topologies}) == 1
        and len({row.device for row in probability_rows}) == 1
        and len({row.dtype for row in probability_rows}) == 1
    )
    if common_shape:
        node_count = len(topologies[0][0])
        max_depth = topologies[0][2]
        all_p_batch = (
            probability_batch
            if isinstance(probability_batch, torch.Tensor)
            else torch.stack(probability_rows)
        )
        if (all_p_batch.ndim != 3 or not all_p_batch.is_contiguous()
                or all_p_batch.shape[:2] != (count, node_count)):
            raise ValueError("Batched sparse probability tensor mismatch")
        metadata = torch.tensor(
            [[parents[1:], tokens]
             for parents, tokens, _, _ in topologies],
            dtype=torch.long, device=all_p_batch.device,
        )
        uniforms = torch.stack([
            torch.rand(
                2 * (max_depth + 1), dtype=torch.float64,
                device=all_p_batch.device, generator=generator,
            )
            for generator in generators
        ])
        packed_rows = (
            extension.fused_tree_sample_sparse_exit_batched_cuda(
                all_p_batch, metadata[:, 0].contiguous(),
                metadata[:, 1].contiguous(), uniforms, max_depth,
            ).tolist()
        )
        results = []
        for (_, tokens, _, vocab), packed in zip(topologies, packed_rows):
            accepted_count = int(packed[max_depth])
            bonus = int(packed[max_depth + 1])
            if not 0 <= accepted_count <= max_depth or not 0 <= bonus < vocab:
                raise RuntimeError("Sparse-exit sampler returned invalid controls")
            nodes = [int(node) for node in packed[:accepted_count]]
            results.append((nodes, [tokens[node - 1] for node in nodes], bonus))
        return results

    packed_outputs = []
    for (parents, tokens, max_depth, _), all_p, generator in zip(
            topologies, probability_rows, generators):
        metadata = torch.tensor(
            [parents[1:], tokens], dtype=torch.long, device=all_p.device,
        )
        uniforms = torch.rand(
            2 * (max_depth + 1), dtype=torch.float64,
            device=all_p.device, generator=generator,
        )
        packed_outputs.append(extension.fused_tree_sample_sparse_exit_cuda(
            all_p, metadata[0], metadata[1], uniforms, max_depth,
        ))

    lengths = [int(output.numel()) for output in packed_outputs]
    host_values = torch.cat(packed_outputs).tolist()
    results = []
    offset = 0
    for (_, tokens, max_depth, vocab), length in zip(topologies, lengths):
        packed = host_values[offset:offset + length]
        offset += length
        accepted_count = int(packed[max_depth])
        bonus = int(packed[max_depth + 1])
        if not 0 <= accepted_count <= max_depth or not 0 <= bonus < vocab:
            raise RuntimeError("Sparse-exit sampler returned invalid control values")
        nodes = [int(node) for node in packed[:accepted_count]]
        results.append((nodes, [tokens[node - 1] for node in nodes], bonus))
    return results


def tree_verify_ancestral_same_draw_fused(
        parents, tokens, all_p, generator=None, validate: bool = True):
    """Use DDTree's exact multinomial draws and fuse only the tree walk.

    This consumes the same random primitive, in the same shape and order, as
    :func:`gbv_experiments.sampling.tree_verify_ancestral_batched`.  Therefore
    a fixed generator state produces identical accepted nodes and bonus token;
    the only change is that traversal happens on CUDA and only the short path
    record is copied to the host.
    """
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2
            or all_p.shape[0] != node_count or all_p.shape[1] < 1
            or not all_p.is_floating_point()):
        raise ValueError("Same-draw DDTree probability tensor mismatch")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for DDTree")

    posterior_tokens = torch.multinomial(
        all_p, 1, generator=generator,
    ).reshape(-1)
    metadata = torch.tensor(
        [parents[1:], tokens], dtype=torch.long, device=all_p.device,
    )
    packed = load_fused_tree_sampler().fused_tree_follow_cuda(
        posterior_tokens, metadata[0], metadata[1], max_depth,
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Same-draw fused tree walk returned invalid controls")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus


def tree_verify_ancestral_logits_fused_scan(
        parents, tokens, all_logits, temperature: float,
        probability_dtype=torch.float64, generator=None,
        validate: bool = True):
    """Normalize and traverse only visited logit rows in one CUDA launch.

    The Target vocabulary head remains unchanged and produces the same BF16
    logits as official DDTree.  The persistent kernel performs stable FP64
    softmax weights, inverse-CDF sampling, and tree traversal without
    materializing a node-by-vocabulary FP64 probability matrix.
    """
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_logits.is_cuda or all_logits.ndim != 2
            or all_logits.shape[0] != node_count
            or all_logits.shape[1] < 1
            or not all_logits.is_floating_point()
            or not all_logits.is_contiguous()):
        raise ValueError("Direct-logits Target tensor mismatch")
    if (not isinstance(temperature, (float, int)) or temperature <= 0
            or not math.isfinite(float(temperature))):
        raise ValueError("Direct-logits temperature must be finite and positive")
    if probability_dtype != torch.float64:
        raise ValueError("Direct-logits verifier requires FP64 probability arithmetic")
    if any(token < 0 or token >= all_logits.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate and not bool(torch.isfinite(all_logits).all()):
        raise FloatingPointError("Invalid Target logits for direct-logits DDTree")

    device = all_logits.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator,
    )
    packed = load_fused_tree_sampler().fused_tree_sample_logits_scan_cuda(
        all_logits, edge_parents, edge_tokens, uniforms,
        float(temperature), max_depth,
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_logits.shape[1]:
        raise RuntimeError("Direct-logits sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus, {
        "visited_probability_rows": accepted_count + 1,
        "total_tree_rows": node_count,
        "lm_head_rows": node_count,
    }


def tree_verify_ancestral_logits_fused_scan_batched(
        parents_batch, tokens_batch, all_logits, temperature: float,
        probability_dtype=torch.float64, generators=None,
        validate: bool = True):
    """Traverse equally shaped request trees in one multi-block CUDA launch."""
    count = len(parents_batch)
    if (count < 1 or len(tokens_batch) != count
            or generators is None or len(generators) != count
            or all_logits.ndim != 3 or all_logits.shape[0] != count
            or not all_logits.is_cuda or not all_logits.is_floating_point()
            or not all_logits.is_contiguous()):
        raise ValueError("Batched direct-logits inputs are inconsistent")
    if (not isinstance(temperature, (float, int)) or temperature <= 0
            or not math.isfinite(float(temperature))):
        raise ValueError("Direct-logits temperature must be finite and positive")
    if probability_dtype != torch.float64:
        raise ValueError("Direct-logits verifier requires FP64 probability arithmetic")

    topologies = [_topology(parents, tokens)
                  for parents, tokens in zip(parents_batch, tokens_batch)]
    node_counts = {len(parents) for parents, _, _ in topologies}
    depths = {depth for _, _, depth in topologies}
    if len(node_counts) != 1 or len(depths) != 1:
        raise ValueError("Batched direct-logits trees must share shape and depth")
    node_count = node_counts.pop()
    max_depth = depths.pop()
    if all_logits.shape[1] != node_count or all_logits.shape[2] < 1:
        raise ValueError("Batched direct-logits tensor shape mismatch")
    vocab = all_logits.shape[2]
    if any(token < 0 or token >= vocab
           for _, tokens, _ in topologies for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate and not bool(torch.isfinite(all_logits).all()):
        raise FloatingPointError("Invalid Target logits for direct-logits DDTree")

    metadata = torch.tensor(
        [[parents[1:], tokens] for parents, tokens, _ in topologies],
        dtype=torch.long, device=all_logits.device,
    )
    uniforms = torch.stack([
        torch.rand(
            max_depth + 1, dtype=torch.float64,
            device=all_logits.device, generator=generator,
        )
        for generator in generators
    ])
    packed_batch = (
        load_fused_tree_sampler().fused_tree_sample_logits_scan_batched_cuda(
            all_logits, metadata[:, 0].contiguous(),
            metadata[:, 1].contiguous(), uniforms,
            float(temperature), max_depth,
        ).tolist()
    )
    results = []
    for (_, tokens, _), packed in zip(topologies, packed_batch):
        accepted_count = int(packed[max_depth])
        bonus = int(packed[max_depth + 1])
        if not 0 <= accepted_count <= max_depth or not 0 <= bonus < vocab:
            raise RuntimeError("Batched direct-logits sampler returned invalid control")
        nodes = [int(node) for node in packed[:accepted_count]]
        results.append((
            nodes, [tokens[node - 1] for node in nodes], bonus,
            {"visited_probability_rows": accepted_count + 1,
             "total_tree_rows": node_count, "lm_head_rows": node_count},
        ))
    return results


def tree_verify_ancestral_logits_sparse_exit_batched(
        parents_batch, tokens_batch, all_logits, log_normalizers,
        temperature: float, probability_dtype=torch.float64,
        generators=None, validate: bool = True):
    """Route on sparse logit events and scan only each request's exit row."""
    count = len(parents_batch)
    if (count < 1 or len(tokens_batch) != count
            or generators is None or len(generators) != count
            or all_logits.ndim != 3 or all_logits.shape[0] != count
            or not all_logits.is_cuda or not all_logits.is_floating_point()
            or not all_logits.is_contiguous()
            or log_normalizers.shape != all_logits.shape[:2]
            or log_normalizers.device != all_logits.device
            or log_normalizers.dtype != torch.float64
            or not log_normalizers.is_contiguous()):
        raise ValueError("Batched sparse-logit inputs are inconsistent")
    if (not isinstance(temperature, (float, int)) or temperature <= 0
            or not math.isfinite(float(temperature))):
        raise ValueError("Sparse-logit temperature must be finite and positive")
    if probability_dtype != torch.float64:
        raise ValueError("Sparse-logit verification requires FP64 normalization")

    topologies = [_topology(parents, tokens)
                  for parents, tokens in zip(parents_batch, tokens_batch)]
    node_counts = {len(parents) for parents, _, _ in topologies}
    depths = {depth for _, _, depth in topologies}
    if len(node_counts) != 1 or len(depths) != 1:
        raise ValueError("Batched sparse-logit trees must share shape and depth")
    node_count = node_counts.pop()
    max_depth = depths.pop()
    if node_count > 65 or all_logits.shape[1] != node_count:
        raise ValueError("Sparse-logit verification supports at most 64 edges")
    vocab = all_logits.shape[2]
    if any(token < 0 or token >= vocab
           for _, tokens, _ in topologies for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_logits).all()
                 & torch.isfinite(log_normalizers).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target logits or log normalizers")

    metadata = torch.tensor(
        [[parents[1:], tokens] for parents, tokens, _ in topologies],
        dtype=torch.long, device=all_logits.device,
    )
    uniforms = torch.stack([
        torch.rand(
            2 * (max_depth + 1), dtype=torch.float64,
            device=all_logits.device, generator=generator,
        )
        for generator in generators
    ])
    packed_batch = (
        load_fused_tree_sampler().fused_tree_sample_logits_sparse_exit_batched_cuda(
            all_logits, log_normalizers,
            metadata[:, 0].contiguous(), metadata[:, 1].contiguous(),
            uniforms, float(temperature), max_depth,
        ).tolist()
    )
    results = []
    for (_, tokens, _), packed in zip(topologies, packed_batch):
        accepted_count = int(packed[max_depth])
        bonus = int(packed[max_depth + 1])
        if not 0 <= accepted_count <= max_depth or not 0 <= bonus < vocab:
            raise RuntimeError("Sparse-logit sampler returned invalid controls")
        nodes = [int(node) for node in packed[:accepted_count]]
        results.append((nodes, [tokens[node - 1] for node in nodes], bonus))
    return results


def _sample_row_inverse_cdf(probability_row, uniform):
    """Use the same FP64 inverse-CDF rule as the persistent scan kernel."""
    cumulative = probability_row.cumsum(0)
    threshold = uniform * cumulative[-1]
    token = torch.searchsorted(cumulative, threshold, right=False)
    return int(token.clamp_max(probability_row.numel() - 1).item())


def _compact_tree_metadata(parents, tokens, internal_nodes, device):
    node_count = len(parents)
    internal_map = [-1] * node_count
    for compact_row, node in enumerate(internal_nodes):
        internal_map[node] = compact_row
    if internal_map[0] != 0:
        raise ValueError("A nontrivial probability tree must have an internal root")
    # Pack every integer control array into one small host-to-device transfer.
    # Row 3 stores the compact-to-original map and is padded only so the tensor
    # remains rectangular; the kernel never reads that padding.
    return torch.tensor([
        internal_map,
        parents,
        [-1, *tokens],
        [*internal_nodes, *([-1] * (node_count - len(internal_nodes)))],
    ], dtype=torch.long, device=device)


def _tree_verify_internal_fused_scan(
        parents, tokens, max_depth, internal_nodes, metadata, internal_p, uniforms,
        validate: bool = True):
    """Walk compact internal rows and report a leaf that needs one final row."""
    node_count = len(parents)
    internal_nodes = list(internal_nodes)
    if (not internal_p.is_cuda or internal_p.ndim != 2
            or internal_p.shape[0] != len(internal_nodes)
            or internal_p.shape[1] < 1 or not internal_p.is_floating_point()):
        raise ValueError("Compact internal probability tensor mismatch")
    if internal_nodes != sorted(set(parents[1:])) or not internal_nodes:
        raise ValueError("Internal nodes do not match the probability tree")
    if any(token < 0 or token >= internal_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if (not uniforms.is_cuda or uniforms.dtype != torch.float64
            or uniforms.numel() < max_depth + 1):
        raise ValueError("Tree uniforms must be CUDA FP64 and cover the depth")
    if (metadata.shape != (4, node_count) or metadata.dtype != torch.long
            or metadata.device != internal_p.device or not metadata.is_contiguous()):
        raise ValueError("Compact tree metadata tensor mismatch")
    if validate:
        valid = (torch.isfinite(internal_p).all() & (internal_p >= 0).all()
                 & (internal_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid compact internal probabilities")

    packed = load_fused_tree_sampler().fused_internal_tree_sample_scan_cuda(
        internal_p.contiguous(),
        metadata[0], metadata[1, 1:], metadata[2, 1:],
        uniforms,
        max_depth,
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    terminal_leaf = int(packed[max_depth + 2])
    if not 0 <= accepted_count <= max_depth:
        raise RuntimeError("Compact fused tree sampler returned invalid path length")
    if (terminal_leaf < 0) == (bonus < 0):
        raise RuntimeError("Compact fused tree sampler returned an invalid exit")
    if terminal_leaf >= 0 and terminal_leaf in internal_nodes:
        raise RuntimeError("Compact fused tree sampler returned an internal leaf")
    if bonus >= internal_p.shape[1]:
        raise RuntimeError("Compact fused tree sampler returned an invalid bonus")
    nodes = [int(node) for node in packed[:accepted_count]]
    if terminal_leaf >= 0 and (not nodes or nodes[-1] != terminal_leaf):
        raise RuntimeError("Compact fused tree sampler omitted its terminal leaf")
    return nodes, [tokens[node - 1] for node in nodes], bonus, terminal_leaf


def tree_verify_ancestral_lazy_softmax_fused_scan(
        parents, tokens, all_logits, temperature: float,
        probability_dtype=torch.float64, generator=None,
        validate: bool = True):
    """Normalize only internal rows, scan the path, then normalize one leaf.

    The Target transformer and BF16 vocabulary head still run on the exact same
    tree-shaped batch as DDTree.  The optimization is confined to posterior
    construction: leaf softmax rows are deferred until the walk actually reaches
    one, and the internal rows are traversed by the persistent scan kernel.
    """
    from .sampling import probabilities

    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (all_logits.ndim != 2 or all_logits.shape[0] != node_count
            or all_logits.shape[1] < 1 or not all_logits.is_floating_point()):
        raise ValueError("Lazy-softmax Target logits mismatch")
    internal_nodes = sorted(set(parents[1:]))
    if not internal_nodes:
        p = probabilities(all_logits[0], temperature, probability_dtype)
        uniform = torch.rand(
            1, dtype=torch.float64, device=all_logits.device,
            generator=generator,
        )[0]
        bonus = _sample_row_inverse_cdf(p, uniform)
        return [], [], bonus, {
            "internal_projected_rows": 0,
            "leaf_projected_rows": 1,
            "projected_rows": 1,
            "total_tree_rows": 1,
            "lm_head_rows": 1,
            "probability_rows": 1,
        }

    device = all_logits.device
    metadata = _compact_tree_metadata(
        parents, tokens, internal_nodes, device,
    )
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device,
        generator=generator,
    )
    internal_index = metadata[3, :len(internal_nodes)]
    internal_p = probabilities(
        all_logits.index_select(0, internal_index),
        temperature, probability_dtype,
    )
    nodes, output_tokens, bonus, terminal_leaf = (
        _tree_verify_internal_fused_scan(
            parents, tokens, max_depth, internal_nodes, metadata,
            internal_p, uniforms, validate,
        )
    )
    leaf_projected = 0
    if terminal_leaf >= 0:
        leaf_p = probabilities(
            all_logits[terminal_leaf], temperature, probability_dtype,
        )
        if validate:
            valid = (torch.isfinite(leaf_p).all() & (leaf_p >= 0).all()
                     & (leaf_p.sum() > 0))
            if not bool(valid):
                raise FloatingPointError("Invalid lazy-softmax leaf probabilities")
        bonus = _sample_row_inverse_cdf(leaf_p, uniforms[len(nodes)])
        leaf_projected = 1
    return nodes, output_tokens, bonus, {
        "internal_projected_rows": len(internal_nodes),
        "leaf_projected_rows": leaf_projected,
        "projected_rows": len(internal_nodes) + leaf_projected,
        "total_tree_rows": node_count,
        "lm_head_rows": node_count,
        "probability_rows": len(internal_nodes) + leaf_projected,
    }


def tree_verify_ancestral_lazy_projection_fused_scan(
        parents, tokens, final_hidden, lm_head, temperature: float,
        probability_dtype=torch.float64, generator=None,
        validate: bool = True):
    """Project compact internal rows and traverse them with the fused scan.

    This is the more aggressive sibling of lazy-softmax scan: it defers both
    the BF16 vocabulary head and FP64 normalization for all leaves, then applies
    both operations to at most the one reached leaf.
    """
    from .sampling import probabilities

    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (final_hidden.ndim != 2 or final_hidden.shape[0] != node_count
            or not final_hidden.is_floating_point()):
        raise ValueError("Lazy-projection Target hidden states mismatch")
    internal_nodes = sorted(set(parents[1:]))
    if not internal_nodes:
        p = probabilities(
            lm_head(final_hidden[:1])[0], temperature, probability_dtype,
        )
        uniform = torch.rand(
            1, dtype=torch.float64, device=final_hidden.device,
            generator=generator,
        )[0]
        bonus = _sample_row_inverse_cdf(p, uniform)
        return [], [], bonus, {
            "internal_projected_rows": 0,
            "leaf_projected_rows": 1,
            "projected_rows": 1,
            "total_tree_rows": 1,
            "lm_head_rows": 1,
            "probability_rows": 1,
        }

    device = final_hidden.device
    metadata = _compact_tree_metadata(
        parents, tokens, internal_nodes, device,
    )
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device,
        generator=generator,
    )
    internal_logits = lm_head(final_hidden.index_select(
        0, metadata[3, :len(internal_nodes)],
    ))
    internal_p = probabilities(
        internal_logits, temperature, probability_dtype,
    )
    nodes, output_tokens, bonus, terminal_leaf = (
        _tree_verify_internal_fused_scan(
            parents, tokens, max_depth, internal_nodes, metadata,
            internal_p, uniforms, validate,
        )
    )
    leaf_projected = 0
    if terminal_leaf >= 0:
        leaf_p = probabilities(
            lm_head(final_hidden[terminal_leaf:terminal_leaf + 1])[0],
            temperature, probability_dtype,
        )
        if validate:
            valid = (torch.isfinite(leaf_p).all() & (leaf_p >= 0).all()
                     & (leaf_p.sum() > 0))
            if not bool(valid):
                raise FloatingPointError(
                    "Invalid lazy-projection fused leaf probabilities"
                )
        bonus = _sample_row_inverse_cdf(leaf_p, uniforms[len(nodes)])
        leaf_projected = 1
    projected_rows = len(internal_nodes) + leaf_projected
    return nodes, output_tokens, bonus, {
        "internal_projected_rows": len(internal_nodes),
        "leaf_projected_rows": leaf_projected,
        "projected_rows": projected_rows,
        "total_tree_rows": node_count,
        "lm_head_rows": projected_rows,
        "probability_rows": projected_rows,
    }


def tree_verify_ancestral_lazy_projection_fused_scan_batched(
        parents_batch, tokens_batch, final_hidden, lm_head,
        temperature: float, probability_dtype=torch.float64,
        generators=None, validate: bool = True, *, sparse_exit: bool = False):
    """Batch compact vocabulary projection and request-local tree walks.

    All internal hidden rows in a Target wave are gathered into one vocabulary
    projection.  One CUDA block then traverses each independent request.  If a
    walk reaches a leaf, all selected leaf rows share a second projection and
    batched inverse-CDF draw.  Request RNG streams and output laws remain
    independent.
    """
    from .sampling import probabilities

    count = len(parents_batch)
    if (count < 1 or len(tokens_batch) != count
            or generators is None or len(generators) != count
            or final_hidden.ndim != 3 or final_hidden.shape[0] != count
            or not final_hidden.is_cuda or not final_hidden.is_floating_point()
            or not final_hidden.is_contiguous()):
        raise ValueError("Batched lazy-projection inputs are inconsistent")
    if (not isinstance(temperature, (float, int)) or temperature <= 0
            or not math.isfinite(float(temperature))):
        raise ValueError("Lazy-projection temperature must be finite and positive")
    if probability_dtype != torch.float64:
        raise ValueError("Lazy projection requires FP64 probability arithmetic")

    topologies = [_topology(parents, tokens)
                  for parents, tokens in zip(parents_batch, tokens_batch)]
    node_counts = {len(parents) for parents, _, _ in topologies}
    depths = {depth for _, _, depth in topologies}
    if len(node_counts) != 1 or len(depths) != 1:
        raise ValueError("Batched lazy-projection trees must share shape and depth")
    node_count = node_counts.pop()
    max_depth = depths.pop()
    if final_hidden.shape[1] != node_count:
        raise ValueError("Batched lazy-projection hidden shape mismatch")
    if sparse_exit and node_count > 65:
        raise ValueError("Sparse lazy projection supports at most 64 tree edges")

    internal_nodes_batch = [
        sorted(set(parents[1:])) for parents, _, _ in topologies
    ]
    if any(not nodes or nodes[0] != 0 for nodes in internal_nodes_batch):
        raise ValueError("Each nontrivial tree must have an internal root")
    internal_capacity = max(map(len, internal_nodes_batch))
    padded_nodes = [
        nodes + [0] * (internal_capacity - len(nodes))
        for nodes in internal_nodes_batch
    ]
    gather_index = torch.tensor(
        [request * node_count + node
         for request, nodes in enumerate(padded_nodes) for node in nodes],
        dtype=torch.long, device=final_hidden.device,
    )
    hidden_size = final_hidden.shape[-1]
    internal_hidden = final_hidden.view(
        count * node_count, hidden_size,
    ).index_select(0, gather_index).view(count, internal_capacity, hidden_size)
    internal_logits = lm_head(internal_hidden)
    internal_p = probabilities(
        internal_logits, temperature, probability_dtype,
    )
    vocab = internal_p.shape[-1]
    if any(token < 0 or token >= vocab
           for _, tokens, _ in topologies for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(internal_p).all() & (internal_p >= 0).all()
                 & (internal_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid batched internal probabilities")

    internal_maps = []
    for nodes in internal_nodes_batch:
        mapping = [-1] * node_count
        for compact, node in enumerate(nodes):
            mapping[node] = compact
        internal_maps.append(mapping)
    internal_rows = torch.tensor(
        internal_maps, dtype=torch.long, device=final_hidden.device,
    )
    edge_parents = torch.tensor(
        [parents[1:] for parents, _, _ in topologies],
        dtype=torch.long, device=final_hidden.device,
    )
    edge_tokens = torch.tensor(
        [tokens for _, tokens, _ in topologies],
        dtype=torch.long, device=final_hidden.device,
    )
    uniforms = torch.stack([
        torch.rand(
            (2 * (max_depth + 1) if sparse_exit else max_depth + 1),
            dtype=torch.float64,
            device=final_hidden.device, generator=generator,
        )
        for generator in generators
    ])
    extension = load_fused_tree_sampler()
    if sparse_exit:
        packed_batch = (
            extension.fused_internal_tree_sample_sparse_exit_batched_cuda(
                internal_p.contiguous(), internal_rows, edge_parents,
                edge_tokens, uniforms, max_depth,
            ).tolist()
        )
    else:
        packed_batch = (
            extension.fused_internal_tree_sample_scan_batched_cuda(
            internal_p.contiguous(), internal_rows, edge_parents,
            edge_tokens, uniforms, max_depth,
            ).tolist()
        )

    parsed = []
    leaf_requests = []
    leaf_nodes = []
    for request, ((_, tokens, _), internal_nodes, packed) in enumerate(zip(
            topologies, internal_nodes_batch, packed_batch)):
        accepted_count = int(packed[max_depth])
        bonus = int(packed[max_depth + 1])
        terminal_leaf = int(packed[max_depth + 2])
        if not 0 <= accepted_count <= max_depth:
            raise RuntimeError("Batched internal sampler returned invalid path length")
        if (terminal_leaf < 0) == (bonus < 0):
            raise RuntimeError("Batched internal sampler returned invalid exit")
        nodes = [int(node) for node in packed[:accepted_count]]
        if terminal_leaf >= 0:
            if terminal_leaf in internal_nodes or not nodes or nodes[-1] != terminal_leaf:
                raise RuntimeError("Batched internal sampler returned invalid leaf")
            leaf_requests.append(request)
            leaf_nodes.append(terminal_leaf)
        elif bonus >= vocab:
            raise RuntimeError("Batched internal sampler returned invalid bonus")
        parsed.append([nodes, [tokens[node - 1] for node in nodes], bonus])

    if leaf_requests:
        leaf_index = torch.tensor(
            [request * node_count + node
             for request, node in zip(leaf_requests, leaf_nodes)],
            dtype=torch.long, device=final_hidden.device,
        )
        leaf_hidden = final_hidden.view(
            count * node_count, hidden_size,
        ).index_select(0, leaf_index)
        leaf_p = probabilities(
            lm_head(leaf_hidden), temperature, probability_dtype,
        )
        leaf_uniform = torch.stack([
            uniforms[
                request,
                (2 * len(parsed[request][0]) + 1
                 if sparse_exit else len(parsed[request][0])),
            ]
            for request in leaf_requests
        ])
        cumulative = leaf_p.cumsum(-1)
        thresholds = leaf_uniform * cumulative[:, -1]
        leaf_bonus = torch.searchsorted(
            cumulative, thresholds[:, None], right=False,
        ).squeeze(1).clamp_max(vocab - 1).tolist()
        for request, bonus in zip(leaf_requests, leaf_bonus):
            parsed[request][2] = int(bonus)

    results = []
    for request, (nodes, output_tokens, bonus) in enumerate(parsed):
        logical_internal = len(internal_nodes_batch[request])
        leaf_projected = int(request in leaf_requests)
        logical_projected = logical_internal + leaf_projected
        results.append((nodes, output_tokens, bonus, {
            "internal_projected_rows": logical_internal,
            "leaf_projected_rows": leaf_projected,
            "projected_rows": logical_projected,
            "total_tree_rows": node_count,
            "lm_head_rows": logical_projected,
            "probability_rows": logical_projected,
            "physical_internal_projection_rows": internal_capacity,
            "sparse_exit": sparse_exit,
        }))
    return results


def tree_verify_ancestral_lazy_projection_sparse_exit_batched(
        parents_batch, tokens_batch, final_hidden, lm_head,
        temperature: float, probability_dtype=torch.float64,
        generators=None, validate: bool = True):
    """Batch lazy vocabulary projection with exact sparse child routing."""
    return tree_verify_ancestral_lazy_projection_fused_scan_batched(
        parents_batch, tokens_batch, final_hidden, lm_head,
        temperature, probability_dtype, generators, validate,
        sparse_exit=True,
    )
