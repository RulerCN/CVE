// CVE.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <chrono>
#include <iostream>
#include <iomanip>

#include "core/core.h"
#include "image/bitmap.h"
#include "nn/mnist.h"
#include "nn/sample_set.h"
#include "nn/linear.h"
#include "nn/sigmoid.h"
#include "nn/softmax.h"

using std::chrono::time_point;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;

//time_point<system_clock> start = system_clock::now();
//time_point<system_clock> stop = system_clock::now();
//long long time = duration_cast<milliseconds>(stop - start).count();

// http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/concepts/library_design/losses.html

// Print vector
template<class Allocator>
void print(const char *name, const core::vector<signed char, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<unsigned char, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<signed short, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<unsigned short, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<int, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<float, Allocator> &vec);
// Print matrix
template<class Allocator>
void print(const char *name, const core::matrix<signed char, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned char, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<signed short, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned short, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<int, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<float, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned long long, Allocator> &mat);

template<class Allocator>
std::ostream& operator<<(std::ostream &os, const core::vector<signed char, Allocator> &v)
{
	std::cout << std::hex << std::setfill('0');
	for (auto i = v.begin(); i != v.end(); ++i)
		std::cout << std::setw(2) << static_cast<short>(static_cast<unsigned char>(*i)) << " ";
	std::cout << std::endl;
	return os;
}

template<class Allocator>
std::ostream& operator<<(std::ostream &os, const core::vector<unsigned char, Allocator> &v)
{
	std::cout << std::hex << std::setfill('0');
	for (auto i = v.begin(); i != v.end(); ++i)
		std::cout << std::setw(2) << static_cast<short>(*i) << " ";
	std::cout << std::endl;
	return os;
}

template<class Allocator>
std::ostream& operator<<(std::ostream &os, const core::matrix<signed char, Allocator> &mat)
{
	std::cout << std::hex << std::setfill('0');
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::setw(2) << static_cast<short>(static_cast<unsigned char>(*i)) << " ";
		std::cout << std::endl;
	}
	return os;
}

template<class Allocator>
std::ostream& operator<<(std::ostream &os, const core::matrix<unsigned char, Allocator> &mat)
{
	std::cout << std::hex << std::setfill('0');
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::setw(2) << static_cast<short>(*i) << " ";
		std::cout << std::endl;
	}
	return os;
}

template<class Allocator>
std::ostream& operator<<(std::ostream &os, const core::tensor<unsigned char, Allocator> &data)
{
	std::cout << std::hex << std::setfill('0');
	for (auto k = data.mbegin(); k != data.mend(); ++k)
	{
		for (auto j = data.vbegin(k); j != data.vend(k); ++j)
		{
			for (auto i = data.begin(j); i != data.end(j); ++i)
				std::cout << std::setw(2) << static_cast<short>(*i) << " ";
		}
		std::cout << std::endl;
	}
	return os;
}

//constexpr double dbl_one    =  1.00000000000000000000e0; // 1
//constexpr double dbl_two    =  2.00000000000000000000e0; // 2
//constexpr double dbl_sqrt2  =  1.41421356237309504880e0; // sqrt(2)
//constexpr double dbl_ln2_hi =  6.9314718246459961e-1;    // ln2 of 20 digit mantissa
//constexpr double dbl_ln2_lo = -1.9046543005827679e-9;    // ln2 - dbl_ln2_hi
//
//constexpr double dbl_log_p0 =  2.00000000000000000000e0;
//constexpr double dbl_log_p1 = -3.56862745098039215686e0;
//constexpr double dbl_log_p2 =  1.95294117647058823529e0;
//constexpr double dbl_log_p3 = -3.33290239172592113769e-1;
//constexpr double dbl_log_p4 =  8.55823914647444059209e-3;
//constexpr double dbl_log_q0 =  1.00000000000000000000e0;
//constexpr double dbl_log_q1 = -2.11764705882352941176e0;
//constexpr double dbl_log_q2 =  1.48235294117647058824e0;
//constexpr double dbl_log_q3 = -3.80090497737556561086e-1;
//constexpr double dbl_log_q4 =  2.59152612093788564377e-2;
//
//constexpr unsigned long long dbl_nan  = 0xfff8000000000000;
//constexpr unsigned long long dbl_inf  = 0x7ff0000000000000;
//constexpr unsigned long long dbl_ninf = 0xfff0000000000000;
//
//double ln(double x)
//{
//	signed long long *i = reinterpret_cast<signed long long*>(&x);
//	// if x < 0 return NAN
//	if (*i & 0x8000000000000000)
//		return *reinterpret_cast<const double*>(&dbl_nan);
//	// if x = 0 return -INF
//	if (x == 0)
//		return *reinterpret_cast<const double*>(&dbl_ninf);
//	// keep the exponent part
//	signed long long exp = ((*i & 0x7ff0000000000000) >> 52) - 1023;
//	double e = static_cast<double>(exp);
//	// keep the decimal part
//	signed long long mant = (*i & 0x000fffffffffffff) | 0x3ff0000000000000;
//	double m = *reinterpret_cast<double*>(&mant);
//	// if m > sqrt(2) e += 1 and m /= 2
//	if (m > dbl_sqrt2)
//	{
//		e += dbl_one;
//		m /= dbl_two;
//	}
//	// t = (m - 1) / (m + 1)
//	double t = (m - dbl_one) / (m + dbl_one);
//	// x = t * t
//	x = t * t;
//	// P(x) = p0 + p1 * x + p2 * x^2 + p3 * x^3 + p4 * x^4
//	double p = dbl_log_p4;
//	p = p * x + dbl_log_p3;
//	p = p * x + dbl_log_p2;
//	p = p * x + dbl_log_p1;
//	p = p * x + dbl_log_p0;
//	// Q(x) = q0 + q1 * x + q2 * x^2 + q3 * x^3 + q4 * x^4
//	double q = dbl_log_q4;
//	q = q * x + dbl_log_q3;
//	q = q * x + dbl_log_q2;
//	q = q * x + dbl_log_q1;
//	q = q * x + dbl_log_q0;
//	// y = t * P(x) / Q(x)
//	double y = t * p / q;
//	// y += e * ln2;
//	y += e * dbl_ln2_hi;
//	y += e * dbl_ln2_lo;
//	return y;
//}

int main()
{
	//static constexpr double dbl_nan = std::numeric_limits<double>::signaling_NaN();
	//static constexpr double dbl_inf = std::numeric_limits<double>::infinity();

	//bool isnan = std::isnan(dbl_nan);
	//bool isinf = std::isinf(dbl_inf);

	//std::uint64_t f1n;
	//std::memcpy(&f1n, &nan, sizeof inf);
	//std::cout << "nan(\"0\") = " << inf << " (" << std::hex << f1n << ")\n";

	float small = core::flt_epsilon;

	float one1 = core::flt_one;
	float one2 = core::flt_one + core::flt_epsilon;
	float max1 = one1 / one2;
	float max2 = max1 + core::flt_epsilon;

	float min1 = core::flt_one / core::flt_epsilon - core::flt_one;

	float sigmoid = core::flt_one / (core::flt_one + min1);


	// 1.192092896e-07 / (1 - 1.192092896e-07)
	// 1.192092896e-07

	// 2.2204460492503131e-016 / (1 - 2.2204460492503131e-016)
	// 2.2204460492503131e-016

	float min_x =-15.942385033256568611253644262487F;
	float max_x = 15.942385152465865316681572419504F;

	double sigmoid_min = -3.60436533891171558590e001;
	double sigmoid_max =  3.60436533891171560811e001;

	float min_exp = core::exp(min_x);
	float max_exp = core::exp(max_x);

	float min_s = core::flt_one / (core::flt_one + min_exp);
	float max_s = core::flt_one / (core::flt_one + max_exp);

	int none1 = *((int*)&one1);
	int none2 = *((int*)&one2);
	int nmax1 = *((int*)&max1);
	int nmax2 = *((int*)&max2);

	std::cout << std::hex << none1 << "\n";
	std::cout << std::hex << none2 << "\n";
	std::cout << std::hex << nmax1 << "\n";
	std::cout << std::hex << nmax2 << "\n";

	float x = -1.25F;
	float y1 = core::exp(x);
	float y2 = exp(x);
	__m128 xmm_y = core::expf4<core::cpu_sse2 | core::cpu_fma>(_mm_set1_ps(x));
	__m256 ymm_y = core::expf8<core::cpu_avx | core::cpu_fma>(_mm256_set1_ps(x));
	std::cout << y1 << "\n";
	std::cout << y2 << "\n";

 	const size_t batch_size = 16;
	const size_t sample_size = 1000;
	nn::sample_set<float, unsigned char> batch_samples(batch_size, 1, 1, 2);
	nn::sample_set<float, unsigned char> train_samples(sample_size, 1, 1, 2);

	// Assign random value
	std::default_random_engine engine(1U);
	std::normal_distribution<float> distribution(0.0F, 0.08F);
	float *pData = train_samples.data.data();
	unsigned char *pLabel = train_samples.labels.data();
	for (size_t i = 0; i < sample_size; i += 2)
	{
		*pData++ = (float)distribution(engine) - 0.2F;
		*pData++ = (float)distribution(engine) - 0.1F;
		*pLabel++ = 0;
	}
	for (size_t i = 0; i < sample_size; i += 2)
	{
		*pData++ = (float)distribution(engine) + 0.2F;
		*pData++ = (float)distribution(engine) + 0.1F;
		*pLabel++ = 1;
	}

	//// Save train samples as image
	//float scale = 320.0F;
	//core::matrix<unsigned char> train_image(320, 320, 3, (unsigned char)255);
	//pData = train_samples.data.data();
	//pLabel = train_samples.labels.data();
	//for (size_t i = 0; i < sample_size; ++i)
	//{
	//	int x = 160 + static_cast<int>(*pData++ * scale);
	//	int y = 160 + static_cast<int>(*pData++ * scale);
	//	if (x >= 0 && x < 320 && y >= 0 && y < 320)
	//	{
	//		unsigned char value = *pLabel++ ? 0xFF : 0x00;
	//		train_image[y][x][0] = 255 - value;
	//		train_image[y][x][1] = 0;
	//		train_image[y][x][2] = value;
	//	}
	//}
	//img::bitmap::encode("data/train_samples.bmp", train_image);

	const size_t in_dim = 2;
	const size_t hide_dim = 5;
	const size_t out_dim = 2;
	const float rate = 0.01F;
	nn::linear<float> layer1(in_dim, hide_dim, true, rate, 0.0F, 0.01F);
	nn::sigmoid<float> layer2;
	nn::linear<float> layer3(hide_dim, out_dim, true, rate, 0.0F, 0.01F);
	nn::sigmoid<float> layer4;

	train_samples.shuffle(1U);
	train_samples.next_batch(batch_samples);

	for (size_t loop = 0; loop < 1; ++loop)
	{
		core::tensor<float> &tensor1 = layer1.forward(batch_samples.data);
		core::tensor<float> &tensor2 = layer2.forward(tensor1);
		core::tensor<float> &tensor3 = layer3.forward(tensor2);
		core::tensor<float> &tensor4 = layer4.forward(tensor3);
		//core::tensor<float> &loss = layer4.backward(batch_samples.labels);
	}

	//__m256d ymm_t0 = _mm256_set_pd(1, 2, 3, 4);
	//__m256d ymm_a0, ymm_a1;

	//ymm_a0 = _mm256_permute2f128_pd(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 2, 0, 0));
	//ymm_a1 = _mm256_permute2f128_pd(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 3, 0, 1));
	//ymm_t0 = _mm256_min_pd(ymm_a0, ymm_a1);
	//ymm_a0 = _mm256_shuffle_pd(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 0, 0, 0));
	//ymm_a1 = _mm256_shuffle_pd(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 0, 3, 3));
	//ymm_t0 = _mm256_min_pd(ymm_a0, ymm_a1);

	//ymm_a0 = ymm_t0;
	//__m128d xmm_t0 = _mm_set_pd(1, 2);
	//__m128d xmm_a0, xmm_a1;
	//xmm_a0 = _mm_shuffle_pd(xmm_t0, xmm_t0, _MM_SHUFFLE(0, 0, 0, 0));
	//xmm_a1 = _mm_shuffle_pd(xmm_t0, xmm_t0, _MM_SHUFFLE(0, 0, 3, 3));
	//xmm_t0 = _mm_min_pd(xmm_a0, xmm_a1);

	//__m128 xmm_t0 = _mm_set_ps(1, 2, 3, 4);
	//__m128 xmm_a0, xmm_a1;

	//xmm_a0 = _mm_shuffle_ps(xmm_t0, xmm_t0, _MM_SHUFFLE(1, 0, 1, 0));
	//xmm_a1 = _mm_shuffle_ps(xmm_t0, xmm_t0, _MM_SHUFFLE(3, 2, 3, 2));
	//xmm_t0 = _mm_min_ps(xmm_a0, xmm_a1);
	//xmm_a0 = _mm_shuffle_ps(xmm_t0, xmm_t0, _MM_SHUFFLE(2, 0, 2, 0));
	//xmm_a1 = _mm_shuffle_ps(xmm_t0, xmm_t0, _MM_SHUFFLE(3, 1, 3, 1));
	//xmm_t0 = _mm_min_ps(xmm_a0, xmm_a1);

	//	xmm_t0 = _mm_min_epi8(xmm_t0, xmm_t4);
	//	xmm_t1 = _mm_min_epi8(xmm_t1, xmm_t5);
	//	xmm_t2 = _mm_min_epi8(xmm_t2, xmm_t6);
	//	xmm_t3 = _mm_min_epi8(xmm_t3, xmm_t7);

	//	xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
	//	xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
	//	xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
	//	xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
	//	xmm_a0 = _mm_min_epi8(xmm_a0, xmm_a2);
	//	xmm_a1 = _mm_min_epi8(xmm_a1, xmm_a3);
	//	xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
	//	xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
	//	xmm_t0 = _mm_min_epi8(xmm_t0, xmm_t1);

	//// 120 376
	//const signed char ptr_a0[] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
	//const signed char ptr_a1[] = { 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f };
	//const signed char ptr_a2[] = { 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f };
	//const signed char ptr_a3[] = { 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f };
	//const signed char ptr_a4[] = { 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f };
	//const signed char ptr_a5[] = { 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f };
	//const signed char ptr_a6[] = { 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f };
	//const signed char ptr_a7[] = { 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f };
	//const signed char ptr_a8[] = { 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f };
	//const signed char ptr_a9[] = { 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f };
	//const signed char ptr_aa[] = { 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf };
	//const signed char ptr_ab[] = { 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf };
	//const signed char ptr_ac[] = { 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf };
	//const signed char ptr_ad[] = { 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf };
	//const signed char ptr_ae[] = { 0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef };
	//const signed char ptr_af[] = { 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff };
	//__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
	//__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
	//// load data from memory
	//xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0));
	//xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1));
	//xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2));
	//xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3));
	//xmm_a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4));
	//xmm_a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5));
	//xmm_a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6));
	//xmm_a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7));
	//xmm_a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8));
	//xmm_a9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9));
	//xmm_aa = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa));
	//xmm_ab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab));
	//xmm_ac = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac));
	//xmm_ad = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad));
	//xmm_ae = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae));
	//xmm_af = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af));
	//// data-type conversion
	//ymm_a0 = _mm256_cvtepu8_epi16(xmm_a0);
	//ymm_a1 = _mm256_cvtepu8_epi16(xmm_a1);
	//ymm_a2 = _mm256_cvtepu8_epi16(xmm_a2);
	//ymm_a3 = _mm256_cvtepu8_epi16(xmm_a3);
	//ymm_a4 = _mm256_cvtepu8_epi16(xmm_a4);
	//ymm_a5 = _mm256_cvtepu8_epi16(xmm_a5);
	//ymm_a6 = _mm256_cvtepu8_epi16(xmm_a6);
	//ymm_a7 = _mm256_cvtepu8_epi16(xmm_a7);
	//ymm_a8 = _mm256_cvtepu8_epi16(xmm_a8);
	//ymm_a9 = _mm256_cvtepu8_epi16(xmm_a9);
	//ymm_aa = _mm256_cvtepu8_epi16(xmm_aa);
	//ymm_ab = _mm256_cvtepu8_epi16(xmm_ab);
	//ymm_ac = _mm256_cvtepu8_epi16(xmm_ac);
	//ymm_ad = _mm256_cvtepu8_epi16(xmm_ad);
	//ymm_ae = _mm256_cvtepu8_epi16(xmm_ae);
	//ymm_af = _mm256_cvtepu8_epi16(xmm_af);
	//// return the horizontal sum
	//ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a1);
	//ymm_a2 = _mm256_hadd_epi16(ymm_a2, ymm_a3);
	//ymm_a4 = _mm256_hadd_epi16(ymm_a4, ymm_a5);
	//ymm_a6 = _mm256_hadd_epi16(ymm_a6, ymm_a7);
	//ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_a9);
	//ymm_aa = _mm256_hadd_epi16(ymm_aa, ymm_ab);
	//ymm_ac = _mm256_hadd_epi16(ymm_ac, ymm_ad);
	//ymm_ae = _mm256_hadd_epi16(ymm_ae, ymm_af);
	//ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a2);
	//ymm_a4 = _mm256_hadd_epi16(ymm_a4, ymm_a6);
	//ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_aa);
	//ymm_ac = _mm256_hadd_epi16(ymm_ac, ymm_ae);
	//ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a4);
	//ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_ac);
	//ymm_a1 = _mm256_permute2f128_si256(ymm_a0, ymm_a8, _MM_SHUFFLE(0, 2, 0, 0));
	//ymm_a9 = _mm256_permute2f128_si256(ymm_a0, ymm_a8, _MM_SHUFFLE(0, 3, 0, 1));
	//ymm_a0 = _mm256_add_epi16(ymm_a1, ymm_a9);

	//__m256 ymm_c0 = _mm256_set_ps(.0008f, .0007f, .0006f, .0005f, .0004f, .0003f, .0002f, .0001f);
	//__m256 ymm_c1 = _mm256_set_ps(.008f, .007f, .006f, .005f, .004f, .003f, .002f, .001f);
	//__m256 ymm_c2 = _mm256_set_ps(.08f, .07f, .06f, .05f, .04f, .03f, .02f, .01f);
	//__m256 ymm_c3 = _mm256_set_ps(.8f, .7f, .6f, .5f, .4f, .3f, .2f, .1f);
	//__m256 ymm_c4 = _mm256_set_ps(8, 7, 6, 5, 4, 3, 2, 1);
	//__m256 ymm_c5 = _mm256_set_ps(80, 70, 60, 50, 40, 30, 20, 10);
	//__m256 ymm_c6 = _mm256_set_ps(800, 700, 600, 500, 400, 300, 200, 100);
	//__m256 ymm_c7 = _mm256_set_ps(8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000);
	//ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c1);
	//ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c3);
	//ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c5);
	//ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c7);
	//ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c2);
	//ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c6);
	//ymm_c1 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 2, 0, 0));
	//ymm_c5 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 3, 0, 1));
	//ymm_c0 = _mm256_add_ps(ymm_c1, ymm_c5);

	//__m256i ymm_c0 = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
	//__m256i ymm_c1 = _mm256_set_epi32(80, 70, 60, 50, 40, 30, 20, 10);
	//__m256i ymm_c2 = _mm256_set_epi32(800, 700, 600, 500, 400, 300, 200, 100);
	//__m256i ymm_c3 = _mm256_set_epi32(8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000);
	//__m256i ymm_c4 = _mm256_set_epi32(80000, 70000, 60000, 50000, 40000, 30000, 20000, 10000);
	//__m256i ymm_c5 = _mm256_set_epi32(800000, 700000, 600000, 500000, 400000, 300000, 200000, 100000);
	//__m256i ymm_c6 = _mm256_set_epi32(8000000, 7000000, 6000000, 5000000, 4000000, 3000000, 2000000, 1000000);
	//__m256i ymm_c7 = _mm256_set_epi32(80000000, 70000000, 60000000, 50000000, 40000000, 30000000, 20000000, 10000000);
	//ymm_c0 = _mm256_hadd_epi32(ymm_c0, ymm_c1);
	//ymm_c2 = _mm256_hadd_epi32(ymm_c2, ymm_c3);
	//ymm_c4 = _mm256_hadd_epi32(ymm_c4, ymm_c5);
	//ymm_c6 = _mm256_hadd_epi32(ymm_c6, ymm_c7);
	//ymm_c0 = _mm256_hadd_epi32(ymm_c0, ymm_c2);
	//ymm_c4 = _mm256_hadd_epi32(ymm_c4, ymm_c6);
	//ymm_c1 = _mm256_permute2f128_si256(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 2, 0, 0));
	//ymm_c5 = _mm256_permute2f128_si256(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 3, 0, 1));
	//ymm_c0 = _mm256_add_epi32(ymm_c1, ymm_c5);

	//try
	//{
	//	std::string input_image = "data/test.bmp";
	//	std::string conv_image = "data/conv.bmp";

	//	img::bitmap_palette palette;
	//	core::matrix<unsigned char> img_input;
	//	if (img::bitmap::decode(input_image, img_input, palette))
	//	{
	//		size_t input_h = img_input.rows();
	//		size_t input_w = img_input.columns();
	//		size_t channels = img_input.dimension();
	//		size_t window_h = 3;
	//		size_t window_w = 3;
	//		size_t stride_h = 1;
	//		size_t stride_w = 1;
	//		size_t output_h = (input_h - window_h) / stride_h + 1;
	//		size_t output_w = (input_w - window_w) / stride_w + 1;
	//	//	core::matrix<float>  mat_kernel(1, window_h * window_w, 1);
	//		core::vector<float>  vec_kernel(window_h * window_w, 1);
	//		core::matrix<float>  mat_image(input_h, input_w, channels);
	//		core::matrix<size_t> mat_index(output_h * output_w * channels, window_h * window_w, 1);
	//		core::matrix<float>  mat_input(output_h * output_w * channels, window_h * window_w, 1);
	//	//	core::matrix<float>  mat_output(output_h * output_w * channels, 1, 1);
	//	//	core::matrix<unsigned char> img_output(output_h, output_w, channels);
	//		core::vector<float>  vec_output(output_h * output_w * channels, 1, 0.0F);
	//		core::vector<unsigned char> vec_matrix(output_h * output_w * channels, 1);
	//		core::matrix<unsigned char> img_output(output_h, output_w, channels, vec_matrix.data());

	//		//mat_kernel.fill({
	//		//	0.0625F, 0.1250F, 0.0625F,
	//		//	0.1250F, 0.2500F, 0.1250F,
	//		//	0.0625F, 0.1250F, 0.0625F
	//		//});

	//		//mat_kernel.fill({
	//		//	1.0F, 0.0F, -1.0F,
	//		//	2.0F, 0.0F, -2.0F,
	//		//	1.0F, 0.0F, -1.0F
	//		//});

	//		vec_kernel.fill({
	//			1.0F, 2.0F, 1.0F,
	//			0.0F, 0.0F, 0.0F,
	//			-1.0F, -2.0F, -1.0F
	//		});

	//		time_point<system_clock> time0 = system_clock::now();
	//		core::cpu_convert(mat_image, img_input);
	//		time_point<system_clock> time1 = system_clock::now();
	//		core::cpu_sliding_window(mat_index, input_h, input_w, channels, window_h, window_w, stride_h, stride_w);
	//		time_point<system_clock> time2 = system_clock::now();
	//		core::cpu_mapping(mat_input, mat_image.data(), mat_index);
	//		time_point<system_clock> time3 = system_clock::now();
	//	//	core::cpu_multiply(mat_output, mat_input, mat_kernel, true);
	//		core::cpu_mul(vec_output, mat_input, vec_kernel);

	//		time_point<system_clock> time4 = system_clock::now();
	//	//	core::cpu_convert(img_output, mat_output);
	//		core::cpu_convert(vec_matrix, vec_output);
	//		time_point<system_clock> time5 = system_clock::now();

	//		long long _time0 = duration_cast<milliseconds>(time5 - time0).count();
	//		long long _time1 = duration_cast<milliseconds>(time1 - time0).count();
	//		long long _time2 = duration_cast<milliseconds>(time2 - time1).count();
	//		long long _time3 = duration_cast<milliseconds>(time3 - time2).count();
	//		long long _time4 = duration_cast<milliseconds>(time4 - time3).count();
	//		long long _time5 = duration_cast<milliseconds>(time5 - time4).count();

	//		std::cout << "total              " << _time0 << " ms" << std::endl;
	//		std::cout << "cpu_convert        " << _time1 << " ms" << std::endl;
	//		std::cout << "cpu_sliding_window " << _time2 << " ms" << std::endl;
	//		std::cout << "cpu_mapping        " << _time3 << " ms" << std::endl;
	//		std::cout << "cpu_multiply       " << _time4 << " ms" << std::endl;
	//		std::cout << "cpu_convert        " << _time5 << " ms" << std::endl;

	//		img::bitmap::encode(conv_image, img_output);
	//	}
	//	else
	//		std::cout << "Can't load image file '" << input_image.data() << "'." << std::endl;
	//}
	//catch (std::exception err)
	//{
	//	std::cout << err.what() << std::endl;
	//}
	//return 0;

	

	//std::cout << "mat_mul(): " << std::endl;
	//for (int i = 64; i <= 1024 * 2; i += 64)
	//{
	//	const size_t m = i;
	//	const size_t n = i;
	//	const size_t k = i;
	//	const size_t d = 1;
	//	core::matrix<float> a(m, k, d);
	//	core::matrix<float> b(k, n, d);
	//	core::matrix<float> c(m, n, d);
	//	a.fill(1.1f);
	//	b.fill(1.2f);

	//	time_point<system_clock> start = system_clock::now();
	//	core::cpu_mul(c, a, b, true);
	//	time_point<system_clock> stop = system_clock::now();
	//	long long time = duration_cast<milliseconds>(stop - start).count();
	//	std::cout << i << ".\t" << m * n * k / 1073741824.0 * (2000.0 / time) << " FLOPS" << std::endl;
	//}
	//std::cout << "OK" << std::endl;
	//return 0;

	try
	{
		core::cpu_inst::enable_simd(true);

		size_t row = 35;
		size_t col = 35;
		size_t dim = 1;
		core::matrix<float> a(row, col, dim);
		core::vector<float> min(row, dim);
		core::vector<float> mean(row, dim);
		core::vector<float> max(row, dim);
		core::vector<float> mint(col, dim);
		core::vector<float> meant(col, dim);
		core::vector<float> maxt(col, dim);

		// Initialization
		core::cpu_bilinear(a, 0.0F, 0.1F, 1.0F);
		core::cpu_min(min, a, core::axis_x);
		core::cpu_mean(mean, a, core::axis_x);
		core::cpu_max(max, a, core::axis_x);
		core::cpu_min(mint, a, core::axis_y);
		core::cpu_mean(meant, a, core::axis_y);
		core::cpu_max(maxt, a, core::axis_y);

		print("a", a);
		print("min", min);
		print("mean", mean);
		print("max", max);
		print("mint", mint);
		print("meant", meant);
		print("maxt", maxt);

		core::matrix<float> vv(col, col, dim);
		core::cpu_gtvv(vv, mint, maxt);
		print("gtvv", vv);

		//size_t row = 23;
		//size_t p = 25;
		//size_t col = 26;
		//size_t dim = 1;
		//core::matrix<double> a(row, p, dim);
		//core::matrix<double> b(p, col, dim);
		//core::matrix<double> t(col, p, dim);
		//core::matrix<double> c(row, col, dim);
		//core::matrix<double> d(row, col, dim);

		//core::matrix<float> fa(row, p, dim);
		//core::matrix<float> fb(p, col, dim);
		//core::matrix<float> fc(row, col, dim);
		//core::matrix<float> fd(row, col, dim);
		//core::matrix<float> fe(row, col, dim);

		//// Initialization matrix
		//a.linear_fill(1.0F, 1.0F);
		//b.linear_fill(1.0F, 1.0F);

		//core::cpu_transpose(t, b);
		//// Matrix-matrix multiplication
		//core::cpu_gemm(c, a, b);
		//core::cpu_gemm(d, a, t, true);

		//core::cpu_convert(fa, a);
		//core::cpu_convert(fb, b);
		//core::cpu_convert(fc, c);
		//core::cpu_convert(fd, d);

		//core::cpu_sub(fe, fc, fd);

		//print("a", fa);
		//print("b", fb);
		//print("c", fc);
		//print("d", fd);
		//print("e", fe);
	}
	catch (std::exception err)
	{
		std::cout << err.what() << std::endl;
	}
	return 0;

	//try
	//{
	//	std::string input_image = "data/test.bmp";
	//	std::string border_wrap = "data/wrap.bmp";

	//	core::matrix<unsigned char> input;
	//	if (img::bitmap::decode(input_image, input))
	//	{
	//		size_t left = 482 * 2 + 20;
	//		size_t top = 272 * 2 + 20;
	//		size_t right = 482 * 2 + 20;
	//		size_t bottom = 272 * 2 + 20;
	//		size_t rows = input.rows();
	//		size_t columns = input.columns();
	//		size_t dimension = input.dimension();
	//		core::matrix<size_t> index(top + rows + bottom, left + columns + right, dimension);
	//		core::cpu_border(index, left, top, right, bottom, core::border_wrap);

	//		time_point<system_clock> start = system_clock::now();

	//		core::matrix<unsigned char> output(index.rows(), index.columns(), index.dimension());
	//		core::cpu_mapping(output, input.data(), index);

	//		time_point<system_clock> stop = system_clock::now();
	//		long long time = duration_cast<milliseconds>(stop - start).count();
	//		std::cout << time << " ms" << std::endl;

	//		img::bitmap::encode(border_wrap, output);
	//	}
	//	else
	//		std::cout << "Can't load image file '" << input_image.data() << "'." << std::endl;
	//}
	//catch (std::exception err)
	//{
	//	std::cout << err.what() << std::endl;
	//}
	//return 0;

	//std::string input_image = "data/test.bmp";
	//std::string replicate = "data/replicate.bmp";

	//core::matrix<unsigned char> input;
	//if (img::bitmap::decode(input_image, input))
	//{
	//	core::matrix<unsigned char> output(input.rows() * 2, input.columns() * 3, input.dimension());
	//	core::cpu_replicate(output, input, 2, 3);
	//	img::bitmap::encode(replicate, output);
	//}

	//const size_t batch      = 100;
	//const size_t rows       = 28;
	//const size_t columns    = 28;
	//const size_t input_dim  = rows * columns;
	//const size_t hide_dim   = 50;
	//const size_t output_dim = 10;
	//const size_t dimension  = 1;
	//ann::mnist<> mnist("data/mnist");
	//core::tensor<float> train_images(batch, rows, columns, dimension);
	//core::vector<float> train_labels(batch, dimension);

	//ann::linear_layer<float> layer1(input_dim, hide_dim);
	//ann::sigmoid_layer<float> layer2;
	//ann::linear_layer<float> layer3(hide_dim, output_dim);
	//ann::softmax_layer<float> layer4(output_dim);
	//core::tensor<float> tensor1(1, batch, hide_dim, 1);
	//core::tensor<float> tensor2(1, batch, hide_dim, 1);
	//core::tensor<float> tensor3(1, batch, output_dim, 1);
	//core::tensor<float> tensor4(1, batch, output_dim, 1);

	//layer1.initialize_normal(0.F, 0.01F, 1U);
	//layer3.initialize_normal(0.F, 0.01F, 1U);

	//mnist.train.shuffle(1U);
	//mnist.train.next_batch(train_images, train_labels);

	//// Change the shape of the input tensor
	//train_images.reshape(1, batch, train_images.matrix_size(), 1);

	//// linear layer
	//layer1.forward(train_images, tensor1);
	//// sigmoid layer
	//layer2.forward(tensor1, tensor2);
	//// linear layer
	//layer3.forward(tensor2, tensor3);
	//// softmax layer
	//layer4.forward(tensor3, tensor4);

	//print("hide:", layer1[0]);
	//print("hide:", layer2[0]);


	//core::tensor<unsigned char> test_images(batch, rows, columns, dimension);
	//core::vector<unsigned char> test_labels(length, dimension);
	//core::tensor<unsigned char> train_images(batch, rows, columns, dimension);
	//core::cpu_convert(train_images, train_images_flt);
	//img::bitmap::encode("data/train/1.bmp", train_images[0]);
	//img::bitmap::encode("data/train/2.bmp", train_images[1]);
	//img::bitmap::encode("data/train/3.bmp", train_images[2]);
	//img::bitmap::encode("data/train/4.bmp", train_images[3]);
	//img::bitmap::encode("data/train/5.bmp", train_images[4]);
	//img::bitmap::encode("data/train/6.bmp", train_images[5]);
	//img::bitmap::encode("data/train/7.bmp", train_images[6]);
	//img::bitmap::encode("data/train/8.bmp", train_images[7]);
	//mnist.test.shuffle(1U);
	//mnist.test.next_batch(test_images, test_labels);
	//img::bitmap::encode("data/test/1.bmp", test_images[0]);
	//img::bitmap::encode("data/test/2.bmp", test_images[1]);
	//img::bitmap::encode("data/test/3.bmp", test_images[2]);
	//img::bitmap::encode("data/test/4.bmp", test_images[3]);
	//img::bitmap::encode("data/test/5.bmp", test_images[4]);
	//img::bitmap::encode("data/test/6.bmp", test_images[5]);
	//img::bitmap::encode("data/test/7.bmp", test_images[6]);
	//img::bitmap::encode("data/test/8.bmp", test_images[7]);

	return 0;
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed char, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(static_cast<unsigned char>(*i)) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<unsigned char, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*i) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed short, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(static_cast<unsigned short>(*i)) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<unsigned short, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(*i) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed int, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << std::hex << std::setfill('0') << std::setw(8) << *i << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<float, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		//std::cout << std::setfill(' ') << std::setw(6) << *i << ",";
		std::cout << std::setfill(' ') << *i << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<signed char, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(static_cast<unsigned char>(*i)) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned char, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*i) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<signed short, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(static_cast<unsigned short>(*i)) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned short, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(*i) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<int, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::hex << std::setfill('0') << std::setw(8) << *i << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<float, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::setfill(' ') << *i << ",";
			//std::cout << std::setfill(' ') << std::setw(6) << *i << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned long long, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			for (size_t n = 0; n < mat.dimension(); ++n)
				std::cout << std::setfill(' ') << std::setw(3) << i[n] << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}
