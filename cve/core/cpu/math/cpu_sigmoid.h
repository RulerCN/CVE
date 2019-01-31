/*====================================================================
BSD 2-Clause License

Copyright (c) 2018, Ruler
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
====================================================================*/
#pragma once

#ifndef __CORE_CPU_SIGMOID_H__
#define __CORE_CPU_SIGMOID_H__

#include "cpu_exp.h"

namespace core
{
	static constexpr float  flt_sigmoid_min = -1.59423850332565686113e001F; /* ln(epsilon/(1-epslon)) */
	static constexpr float  flt_sigmoid_max =  1.59423851524658653167e001F; /*-ln(epsilon) */
	static constexpr double dbl_sigmoid_min = -3.60436533891171558590e001;  /* ln(epsilon/(1-epslon)) */
	static constexpr double dbl_sigmoid_max =  3.60436533891171560811e001;  /*-ln(epsilon) */

	static constexpr ALIGN(64) float  a8_flt_sigmoid_min[8] = CONST_ARRAY_8(flt_sigmoid_min);
	static constexpr ALIGN(64) float  a8_flt_sigmoid_max[8] = CONST_ARRAY_8(flt_sigmoid_max);
	static constexpr ALIGN(64) double a4_dbl_sigmoid_min[4] = CONST_ARRAY_4(dbl_sigmoid_min);
	static constexpr ALIGN(64) double a4_dbl_sigmoid_max[4] = CONST_ARRAY_4(dbl_sigmoid_max);

	// Function template sigmoid

	template<cpu_inst_type inst>
	float sigmoid(float x)
	{
		if (x < flt_sigmoid_min)
			x = flt_sigmoid_min;
		else if (x > flt_sigmoid_max)
			x = flt_sigmoid_max;
		// y = 1 / (1 + exp(-x));
		float y = flt_one / (flt_one + core::exp(-x));
		return y;
	}

	template<cpu_inst_type inst>
	double sigmoid(double x)
	{
		if (x < dbl_sigmoid_min)
			x = dbl_sigmoid_min;
		else if (x > dbl_sigmoid_max)
			x = dbl_sigmoid_max;
		// y = 1 / (1 + exp(-x));
		double y = dbl_one / (dbl_one + core::exp(-x));
		return y;
	}

	// Function template sigmoidf4

	template<cpu_inst_type inst>
	__m128 sigmoidf4(__m128 xmm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m128 sigmoidf4<cpu_sse2>(__m128 xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sign));
		__m128 xmm_y = core::expf4<cpu_sse2>(xmm_x);
		xmm_y = _mm_add_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		xmm_y = _mm_div_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		return xmm_y;
	}

	template<>
	__m128 sigmoidf4<cpu_sse2 | cpu_fma>(__m128 xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sign));
		__m128 xmm_y = core::expf4<cpu_sse2 | cpu_fma>(xmm_x);
		xmm_y = _mm_add_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		xmm_y = _mm_div_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		return xmm_y;
	}

	template<>
	__m128 sigmoidf4<cpu_sse41>(__m128 xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sign));
		__m128 xmm_y = core::expf4<cpu_sse41>(xmm_x);
		xmm_y = _mm_add_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		xmm_y = _mm_div_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		return xmm_y;
	}

	template<>
	__m128 sigmoidf4<cpu_sse41 | cpu_fma>(__m128 xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_sign));
		__m128 xmm_y = core::expf4<cpu_sse41 | cpu_fma>(xmm_x);
		xmm_y = _mm_add_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		xmm_y = _mm_div_ps(*reinterpret_cast<const __m128*>(a8_flt_one), xmm_y);
		return xmm_y;
	}

	// Function template sigmoidd2

	template<cpu_inst_type inst>
	__m128d sigmoidd2(__m128d xmm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m128d sigmoidd2<cpu_sse2>(__m128d xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sign));
		__m128d xmm_y = core::expd2<cpu_sse2>(xmm_x);
		xmm_y = _mm_add_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		xmm_y = _mm_div_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		return xmm_y;
	}

	template<>
	__m128d sigmoidd2<cpu_sse2 | cpu_fma>(__m128d xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sign));
		__m128d xmm_y = core::expd2<cpu_sse2 | cpu_fma>(xmm_x);
		xmm_y = _mm_add_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		xmm_y = _mm_div_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		return xmm_y;
	}

	template<>
	__m128d sigmoidd2<cpu_sse41>(__m128d xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sign));
		__m128d xmm_y = core::expd2<cpu_sse41>(xmm_x);
		xmm_y = _mm_add_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		xmm_y = _mm_div_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		return xmm_y;
	}

	template<>
	__m128d sigmoidd2<cpu_sse41 | cpu_fma>(__m128d xmm_x)
	{
		// x = max(x, min);
		xmm_x = _mm_max_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_min));
		// x = min(x, max);
		xmm_x = _mm_min_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		xmm_x = _mm_xor_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_sign));
		__m128d xmm_y = core::expd2<cpu_sse41 | cpu_fma>(xmm_x);
		xmm_y = _mm_add_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		xmm_y = _mm_div_pd(*reinterpret_cast<const __m128d*>(a4_dbl_one), xmm_y);
		return xmm_y;
	}

	// Function template sigmoidf8

	template<cpu_inst_type inst>
	__m256 sigmoidf8(__m256 ymm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m256 sigmoidf8<cpu_avx>(__m256 ymm_x)
	{
		// x = max(x, min);
		ymm_x = _mm256_max_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_sigmoid_min));
		// x = min(x, max);
		ymm_x = _mm256_min_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		ymm_x = _mm256_xor_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_sign));
		__m256 ymm_y = core::expf8<cpu_avx>(ymm_x);
		ymm_y = _mm256_add_ps(*reinterpret_cast<const __m256*>(a8_flt_one), ymm_y);
		ymm_y = _mm256_div_ps(*reinterpret_cast<const __m256*>(a8_flt_one), ymm_y);
		return ymm_y;
	}

	template<>
	__m256 sigmoidf8<cpu_avx | cpu_fma>(__m256 ymm_x)
	{
		// x = max(x, min);
		ymm_x = _mm256_max_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_sigmoid_min));
		// x = min(x, max);
		ymm_x = _mm256_min_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		ymm_x = _mm256_xor_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_sign));
		__m256 ymm_y = core::expf8<cpu_avx | cpu_fma>(ymm_x);
		ymm_y = _mm256_add_ps(*reinterpret_cast<const __m256*>(a8_flt_one), ymm_y);
		ymm_y = _mm256_div_ps(*reinterpret_cast<const __m256*>(a8_flt_one), ymm_y);
		return ymm_y;
	}

	// Function template sigmoidd4

	template<cpu_inst_type inst>
	__m256d sigmoidd4(__m256d ymm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m256d sigmoidd4<cpu_avx>(__m256d ymm_x)
	{
		// x = max(x, min);
		ymm_x = _mm256_max_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_sigmoid_min));
		// x = min(x, max);
		ymm_x = _mm256_min_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		ymm_x = _mm256_xor_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_sign));
		__m256d ymm_y = core::expd4<cpu_avx>(ymm_x);
		ymm_y = _mm256_add_pd(*reinterpret_cast<const __m256d*>(a4_dbl_one), ymm_y);
		ymm_y = _mm256_div_pd(*reinterpret_cast<const __m256d*>(a4_dbl_one), ymm_y);
		return ymm_y;
	}

	template<>
	__m256d sigmoidd4<cpu_avx | cpu_fma>(__m256d ymm_x)
	{
		// x = max(x, min);
		ymm_x = _mm256_max_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_sigmoid_min));
		// x = min(x, max);
		ymm_x = _mm256_min_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_sigmoid_max));
		// y = 1 / (1 + exp(-x));
		ymm_x = _mm256_xor_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_sign));
		__m256d ymm_y = core::expd4<cpu_avx | cpu_fma>(ymm_x);
		ymm_y = _mm256_add_pd(*reinterpret_cast<const __m256d*>(a4_dbl_one), ymm_y);
		ymm_y = _mm256_div_pd(*reinterpret_cast<const __m256d*>(a4_dbl_one), ymm_y);
		return ymm_y;
	}

} // namespace core

#endif
