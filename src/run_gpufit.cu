#include "run_gpufit.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

// Run the gpufit 
// N点高斯峰值定位算法(only GPU)
// xx -- (1,n_points), n_points -> number_points, float
// yy0 -- (n_points, h, w), n_fits = h*w ->number_fits, float
// nPoints -- m' points near the peak, Scaler
// sigma_t -- the sigma of gaussian fit, Scaler
std::tuple<torch::Tensor, torch::Tensor> run_gpufit(torch::Tensor& xx, torch::Tensor& yy0, std::size_t nPoints, REAL sigma0,
	REAL tolerance, int max_n_iterations, int model_id, int estimator_id)
{
	auto yy0_size = yy0.sizes();
	int64_t n_points = yy0_size[0];
	int64_t n_fits = yy0_size[1]* yy0_size[2];
	torch::Tensor yy = yy0.permute({ 1,2,0 }).view({ -1, n_points }).contiguous(); // new tensor

	// movemean smooth yy -> yys	
	torch::Tensor yys = torch::avg_pool1d(yy.view({ 1, n_fits, n_points }), /*kernel_size*/5, 1, /*padding*/2);

	// Normalize yy to ones using yys_max
	torch::Tensor yys_max, yys_ind;	
	std::tie(yys_max, yys_ind) = torch::max(yys.view({ n_fits, n_points }), /*dim=*/1, /*keepdim=*/true);
	yy = yy / yys_max;	

	// 估计mu
	torch::Tensor mu = xx.squeeze_().index({ yys_ind });

	// 估计sigma
	torch::Tensor dxx = torch::diff(xx /*n = 1, dim = -1*/);
	torch::Tensor sigma = torch::sum(yy.index({ Ellipsis, Slice(0, n_points - 1) }) * dxx, 1, /*keepdim=*/true) * 0.398942280401433f; // 1/sqrt(2*pi)
	float c = 1.482602218505602f; // -1/(sqrt(2)*erfcinv(3/2))
	torch::Tensor sigma_med = torch::median(sigma);//median
	torch::Tensor sL = sigma < sigma_med; // left
	torch::Tensor distL = c * torch::median(torch::abs(sigma.index({ sL }) - sigma_med));
	torch::Tensor distR = c * torch::median(torch::abs(sigma.index({ ~sL }) - sigma_med));
	torch::Tensor lower = sigma - 3 * sigma_med;
	torch::Tensor upper = sigma + 3 * sigma_med;
	sL = (sigma < lower) | (sigma > upper); // final
	sigma = sigma.clamp_(lower, upper);

	// Perform initial estimate of parameters (极值法)
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
	torch::Tensor fit_parameters = torch::empty({ n_fits,4 }, options);
	fit_parameters.index_put_({ Ellipsis, Slice(0, 1) }, 1); // A
	fit_parameters.index_put_({ Ellipsis, Slice(1, 2) }, mu); // mu
	fit_parameters.index_put_({ Ellipsis, Slice(2, 3) }, sigma); // sigma
	fit_parameters.index_put_({ Ellipsis, Slice(3, 4) }, 0); // offset		

	// 估计weights
	int64_t radius = nPoints / 2;
	torch::Tensor dist = torch::abs(torch::arange(0, n_points, options).view({ 1,-1 }).expand_as(yy) - yys_ind);
	torch::Tensor weights = (dist <= radius).toType(torch::kFloat32); // type promotion, see https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
	/*torch::Tensor lo = (ind - torch::Scalar(radius)) < 1;
	torch::Tensor ro = (ind + torch::Scalar(radius)) > n_points;
	ro.print();*/

	// Pre-allocate tensor	
	auto int_options = torch::dtype(torch::kInt32).device(torch::kCUDA, 0).requires_grad(false);
	torch::Tensor states = torch::empty({n_fits,1}, int_options);
	torch::Tensor chi_squares = torch::empty({ n_fits,1 }, options);
	torch::Tensor n_iterations = torch::empty({ n_fits,1 }, int_options);

	// parameters to fit (fix the 4th parameter)
	std::vector<int> parameters_to_fit{ 1, 1, 1, 0 }; // 不固定offset，收敛效果不好

	// size of user_info in bytes
	size_t const user_info_size = n_points * sizeof(float);

	// call to gpufit_cuda_interface
	int status;		
	status = gpufit_cuda_interface(
		n_fits,
		n_points,
		(float*) yy.data_ptr(),
		(float*) weights.data_ptr(),
		model_id,
		tolerance,
		max_n_iterations,
		parameters_to_fit.data(),
		estimator_id,
		user_info_size,
		(char*) xx.data_ptr(),
		(float*) fit_parameters.data_ptr(),
		(int*) states.data_ptr(),
		(float*) chi_squares.data_ptr(),
		(int*) n_iterations.data_ptr()
	);

    // check status
	if (status != ReturnState::OK)
	{
		throw std::runtime_error(gpufit_get_last_error());
	}

	// 计算置信水平confidenceMap, confidenceMap = 20 * log10(1. / chi_squares);
	// chi_square必须小于: 0.03(≈30.5dB), 0.05(≈26dB), 0.1(≈20dB), 0.15(≈16.5dB)
	float eps = 1e-6;
	torch::Tensor confidenceMap = 20 * log10(1. / (chi_squares + eps));
	torch::Tensor heightMap = fit_parameters.index({ Ellipsis, Slice(1, 2) });

	/*std::cout << states.cpu() << std::endl;
	std::cout << fit_parameters.index({Ellipsis, 1}).cpu() << std::endl;*/

	torch::Tensor tmp = fit_parameters.cpu();
	std::vector<float> aa(tmp.data_ptr<float>(), tmp.data_ptr<float>() + tmp.numel());
	tmp = chi_squares.cpu();
	std::vector<float> bb(tmp.data_ptr<float>(), tmp.data_ptr<float>() + tmp.numel());
	tmp = states.cpu();
	std::vector<int> cc(tmp.data_ptr<int>(), tmp.data_ptr<int>() + tmp.numel());


	// remove outliers
	torch::Tensor L = states != 0; // condition 1
	L = L | (chi_squares > 0.1); // condition 2
	//torch::Tensor L = states != 0;

	confidenceMap.index_put_({ L }, 0);
	heightMap.index_put_({ L }, NAN);

	confidenceMap = confidenceMap.reshape({ yy0_size[1], yy0_size[2] });
	heightMap = heightMap.reshape({ yy0_size[1], yy0_size[2] });

	return std::make_tuple(heightMap, confidenceMap);
}