#pragma once

#include "gpufit.h"
#include <vector>
#include <stdexcept>
#include <algorithm>  // for std::max_element

#include <torch/torch.h>

using namespace torch::indexing;

std::tuple<torch::Tensor, torch::Tensor> run_gpufit(
	torch::Tensor& xx, torch::Tensor& yy0, std::size_t nPoints, REAL sigma0,
	REAL tolerance = 1e-6, int max_n_iterations = 20,
	int model_id = GAUSS_1D, int estimator_id = LSE);