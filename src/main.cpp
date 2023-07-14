// test_cmake_gpufit.cpp: 定义应用程序的入口点。
//

#include "main.h"

//#include <cmath>


#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "run_gpufit.hpp"

//#include <matplot/matplot.h>


//// Function to convert a 2D Tensor to a 2D vector
//std::vector<std::vector<double>> TensorTo2DVector(torch::Tensor tensor) {
//	tensor = tensor.cpu().contiguous();  // ensure tensor is in contiguous memory
//
//	// Get dimensions
//	int64_t rows = tensor.size(0);
//	int64_t cols = tensor.size(1);
//
//	// Create 2D vector of appropriate size
//	std::vector<std::vector<double>> vec(rows, std::vector<double>(cols));
//
//	// Copy tensor to vector
//	auto tensor_accessor = tensor.accessor<float, 2>();
//	for (int64_t i = 0; i < rows; ++i) {
//		for (int64_t j = 0; j < cols; ++j) {
//			vec[i][j] = tensor_accessor[i][j];
//		}
//	}
//
//	return vec;
//}

#include <iomanip>
#include <chrono>

void SaveAsGsf(std::string filename, torch::Tensor data, int numstepsx, int numstepsy, double startx, double endx, double starty, double endy, std::string label, std::string unit, int mtime, int varargin)
{
	int parameterSize = 0;
	FILE* myFile;
	data = data * 1e-6;	
	fopen_s(&myFile, filename.c_str(), "wb+");
	parameterSize += fprintf(myFile, "Gwyddion Simple Field 1.0\n");
	parameterSize += fprintf(myFile, "XRes = %d\n", numstepsx);
	parameterSize += fprintf(myFile, "YRes = %d\n", numstepsy);

	if (startx != -1) {
		parameterSize += fprintf(myFile, "XReal = %.6f\n", abs(endx - startx));
		parameterSize += fprintf(myFile, "YReal = %.6f\n", abs(endy - starty));
		parameterSize += fprintf(myFile, "XOffset = %.6f\n", startx);
		parameterSize += fprintf(myFile, "YOffset = %.6f\n", starty);
	}
	parameterSize += fprintf(myFile, "XYUnits = m\n");

	if (unit != "") {
		parameterSize += fprintf(myFile, "ZUnits = %s\n", unit.c_str());
	}
	if (label != "") {
		parameterSize += fprintf(myFile, "Title = %s\n", label.c_str());
	}
	parameterSize += fprintf(myFile, "Version = Matlab2Gwyddion 1.0\n");

	if (mtime != -1) {
		time_t rawtime;
		struct tm info;
		char buffer[80];

		time(&rawtime);
		localtime_s(&info, &rawtime);
		strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", &info);
		parameterSize += fprintf(myFile, "Date=%s\n", buffer);
	}

	switch (parameterSize % 4)
	{
	case 0:
		fprintf(myFile, "%c%c%c%c", 0, 0, 0, 0);
		break;
	case 1:
		fprintf(myFile, "%c%c%c", 0, 0, 0);
		break;
	case 2:
		fprintf(myFile, "%c%c", 0, 0);
		break;
	case 3:
		fprintf(myFile, "%c", 0);
		break;
	default:
		break;
	}
	fwrite((float*)data.data_ptr(), sizeof(float), numstepsx * numstepsy, myFile);
	fclose(myFile);
	return;
}


using namespace torch::indexing;

int main()
{
	// 读取 TIFF 文件
	std::vector< cv::Mat> img;
	std::vector < cv::cuda::GpuMat > img_gpu;	
	cv::imreadmulti("D:/workspace/chai_project/test_cmake_gpufit/test_sectioning.tif", img, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
	for (int i = 0; i < img.size(); i++)
	{
		cv::cuda::GpuMat gpuImage;
		gpuImage.upload(img[i]);
		img_gpu.emplace_back(gpuImage);		
		/*cv::Mat temp = img[i];
		cv::normalize(temp, temp, 1, 0, cv::NORM_MINMAX);
		cv::imshow("CPU_img", temp);
		cv::waitKey(1);*/
	}
	
	int height = img[0].rows;
	int width = img[0].cols;
	int zN = img.size();    
	int64_t step = img[0].step / (sizeof(float));// prepare for libtorch tensor conversion
	std::vector<int64_t> strides = { step, 1 };
	torch::Deleter deleter;
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
	torch::Tensor yy = torch::empty({ zN, height, width }, options);		
	for (int i = 0; i < zN; i++)
	{
		torch::Tensor tensor_image = torch::from_blob(img_gpu[i].data, { height, width }, strides, deleter, options);
		//tensor_image.print();
		//std::cout << tensor_image.index({ Slice(0, 2, 1), Slice(0, 10, 1) }) << std::endl;
		yy.index_put_({i, Ellipsis}, tensor_image);
		//std::cout << yy.index({i, Slice(0, 2, 1), Slice(0, 10, 1) })<< std::endl;
	}	

	std::chrono::high_resolution_clock::time_point time_0 = std::chrono::high_resolution_clock::now();

	torch::Tensor xx = torch::arange(0, zN, /*step=*/1, options).view({ 1,-1 }) * 0.1f; // (1,zN)
	torch::Tensor heightMap, C;
	std::tie(heightMap, C) = run_gpufit(xx, yy/*.index({ Ellipsis, Slice(0,2), Slice(0,1)})*/, 21, 8);
    
	// print execution time
	std::chrono::high_resolution_clock::time_point time_1 = std::chrono::high_resolution_clock::now();	
	std::cout << "execution time "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count() << " ms" << std::endl;
	
	/*C.print();
	heightMap.print();
	std::cout << heightMap.index({Slice(20, 20+2), Slice(20, 20+10) })<< std::endl;*/

	//plot
	{
		//auto [X, Y] = matplot::meshgrid(matplot::iota(1, width), matplot::iota(1, height));
		//auto Z = TensorTo2DVector(C);
		//ax->surf(X, Y, Z)/*->edge_color("none").lighting(true)*/;
		///*matplot::colormap(matplot::palette::jet());*/
		//matplot::show();
		int obj = 20;
		SaveAsGsf("./results_heightMap.gsf", heightMap.cpu(), width, height, 0, 5.86e-6 * width / obj, 0, 5.86e-6 * height / obj, "Chan1", "m", 0, 0);
		SaveAsGsf("./results_confidenceMap.gsf", C.cpu(), width, height, 0, 5.86e-6 * width / obj, 0, 5.86e-6 * height / obj, "Chan1", "m", 0, 0);
	}


	std::cout << std::endl << "Example completed!" << std::endl;
	return 0;
}


/* cmake cli
cmake -B D:/workspace/chai_project/test_cmake_gpufit/out/build/x64-Debug ^
-S D:/workspace/chai_project/test_cmake_gpufit ^
-G "Ninja"
-D CMAKE_BUILD_TYPE:STRING="Debug"
-D CMAKE_TOOLCHAIN_FILE="D:/packages/common/vcpkg/scripts/buildsystems/vcpkg.cmake"

---------------------- :: debug mode :: --------------------------
cd /d D:/workspace/chai_project/test_cmake_gpufit/out/build/x64-Debug
cmake --build . --parallel 20

copy "C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64\nvToolsExt64_1.dll" .
copy "E:\package\pytorch\build\Debug\bin\nvfuser_codegen.dll" .
---------------------- :: debug mode :: --------------------------


---------------------- :: release mode :: --------------------------
cd /d D:/workspace/chai_project/test_cmake_gpufit/out/build/x64-Release

copy "C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64\nvToolsExt64_1.dll" .
copy "E:\package\pytorch\build\Release\bin\nvfuser_codegen.dll" .
---------------------- :: release mode :: --------------------------

test_cmake_gpufit.exe

*/
