#pragma once

namespace skepu {
namespace filter {


struct GrayscalePixel
{
	unsigned char intensity;
};

struct RGBPixel
{
	unsigned char r, g, b;
};




GrayscalePixel intensity_kernel(RGBPixel input)
{
	GrayscalePixel output;
	output.intensity = ((unsigned int)input.r + (unsigned int)input.g + (unsigned int)input.b) / 3;
	return output;
}

GrayscalePixel convolution_kernel(skepu::Region1D<GrayscalePixel> r, skepu::Vec<float> filter, float offset, float scaling)
{
	GrayscalePixel result;
	float intensity = 0;
	
	for (int i = -r.oi; i <= r.oi; i++)
	{
		GrayscalePixel p = r(i);
		intensity += p.intensity * filter(i+r.oi);
	}
	
	result.intensity = (intensity + offset) * scaling;
	return result;
}

RGBPixel convolution_kernel_rgb(skepu::Region1D<RGBPixel> in, skepu::Vec<float> filter, float offset, float scaling)
{
	RGBPixel result;
	float r = 0;
	float g = 0;
	float b = 0;
	
	
	for (int i = -in.oi; i <= in.oi; i++)
	{
		RGBPixel p = in(i);
		float coeff = filter(i + in.oi);
		r += p.r * coeff;
		g += p.g * coeff;
		b += p.b * coeff;
	}
	
	result.r = (r + offset) * scaling;
	result.g = (g + offset) * scaling;
	result.b = (b + offset) * scaling;
	return result;
}

float gauss_weights_kernel(skepu::Index1D index, size_t r, float sigma)
{
	const float pi = 3.141592;
	float i = (float)index.i - r;
	return exp(-i*i / (2 * sigma * sigma)) / sqrt(2* pi * sigma * sigma);
}

GrayscalePixel distance_kernel(GrayscalePixel x, GrayscalePixel y)
{
	GrayscalePixel result;
	float xf = (float)x.intensity / 255 - 0.5;
	float yf = (float)y.intensity / 255 - 0.5;
	result.intensity = 255 - sqrt(xf*xf + yf*yf) * 2 * 255;
	return result;
}




// Kernel for filter with raduis R
RGBPixel median_kernel(skepu::Region2D<RGBPixel> image)
{
	long fineHistogram[3][256], coarseHistogram[3][16];
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 256; i++)
			fineHistogram[c][i] = 0;
	
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 16; i++)
			coarseHistogram[c][i] = 0;
	
	for (int row = -image.oi; row <= image.oi; row++)
	{
		for (int column = -image.oj; column <= image.oj; column++)
		{ 
			unsigned char imageValue = image(row, column).r;
			fineHistogram[0][imageValue]++;
			coarseHistogram[0][imageValue / 16]++;
			
			imageValue = image(row, column).g;
			fineHistogram[1][imageValue]++;
			coarseHistogram[1][imageValue / 16]++;
			
			imageValue = image(row, column).b;
			fineHistogram[2][imageValue]++;
			coarseHistogram[2][imageValue / 16]++;
		}
	}
	
	unsigned char fineIndex[3];
	
	for (int c = 0; c < 3; c++)
	{
		int count = 2 * image.oi * (image.oi + 1);
		unsigned char coarseIndex;
		for (coarseIndex = 0; coarseIndex < 16; ++coarseIndex)
		{
			if ((long)count - coarseHistogram[c][coarseIndex] < 0) break;
			count -= coarseHistogram[c][coarseIndex];
		}
		
		fineIndex[c] = coarseIndex * 16;
		while ((long)count - fineHistogram[c][fineIndex[c]] >= 0)
			count -= fineHistogram[c][fineIndex[c]++];
	}
	
	RGBPixel res;
	res.r = fineIndex[0];
	res.g = fineIndex[1];
	res.b = fineIndex[2];
	return res;
}

}} // skepu::filter

#ifdef SKEPU_OPENCL

namespace skepu
{
	template<> inline std::string getDataTypeCL<skepu::filter::GrayscalePixel> () { return "struct GrayscalePixel"; }
	template<> inline std::string getDataTypeCL<skepu::filter::RGBPixel> () { return "struct RGBPixel"; }

	template<> inline std::string getDataTypeDefCL<skepu::filter::GrayscalePixel>() { return R"~~~(typedef struct GrayscalePixel
	{
		unsigned char intensity;
	} skepu__colon____colon__filter__colon____colon__GrayscalePixel; typedef struct GrayscalePixel GrayscalePixel;
)~~~"; }
	template<> std::string getDataTypeDefCL<skepu::filter::RGBPixel>() { return R"~~~(typedef struct RGBPixel
	{
		unsigned char r, g, b;
	} skepu__colon____colon__filter__colon____colon__RGBPixel; typedef struct RGBPixel RGBPixel;
)~~~"; }
}

#endif