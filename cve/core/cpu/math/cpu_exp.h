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

#ifndef __CORE_CPU_EXP_H__
#define __CORE_CPU_EXP_H__

#include "../cpu_inst.h"

namespace core
{
	static constexpr float  flt_rcp_fact0  = 1.00000000000000000000e000F; /* 1/0! */
	static constexpr float  flt_rcp_fact1  = 1.00000000000000000000e000F; /* 1/1! */
	static constexpr float  flt_rcp_fact2  = 5.00000000000000000000e-01F; /* 1/2! */
	static constexpr float  flt_rcp_fact3  = 1.66666666666666666667e-01F; /* 1/3! */
	static constexpr float  flt_rcp_fact4  = 4.16666666666666666667e-02F; /* 1/4! */
	static constexpr float  flt_rcp_fact5  = 8.33333333333333333333e-03F; /* 1/5! */
	static constexpr float  flt_rcp_fact6  = 1.38888888888888888889e-03F; /* 1/6! */
	static constexpr float  flt_rcp_fact7  = 1.98412698412698412698e-04F; /* 1/7! */
	static constexpr float  flt_rcp_fact8  = 2.48015873015873015873e-05F; /* 1/8! */
	static constexpr float  flt_rcp_fact9  = 2.75573192239858906526e-06F; /* 1/9! */
	static constexpr float  flt_rcp_fact10 = 2.75573192239858906526e-07F; /* 1/10! */
	static constexpr float  flt_rcp_fact11 = 2.50521083854417187751e-08F; /* 1/11! */
	static constexpr float  flt_rcp_fact12 = 2.08767569878680989792e-09F; /* 1/12! */
	static constexpr float  flt_rcp_fact13 = 1.60590438368216145994e-10F; /* 1/13! */
	static constexpr double dbl_rcp_fact0  = 1.00000000000000000000e000;  /* 1/0! */
	static constexpr double dbl_rcp_fact1  = 1.00000000000000000000e000;  /* 1/1! */
	static constexpr double dbl_rcp_fact2  = 5.00000000000000000000e-01;  /* 1/2! */
	static constexpr double dbl_rcp_fact3  = 1.66666666666666666667e-01;  /* 1/3! */
	static constexpr double dbl_rcp_fact4  = 4.16666666666666666667e-02;  /* 1/4! */
	static constexpr double dbl_rcp_fact5  = 8.33333333333333333333e-03;  /* 1/5! */
	static constexpr double dbl_rcp_fact6  = 1.38888888888888888889e-03;  /* 1/6! */
	static constexpr double dbl_rcp_fact7  = 1.98412698412698412698e-04;  /* 1/7! */
	static constexpr double dbl_rcp_fact8  = 2.48015873015873015873e-05;  /* 1/8! */
	static constexpr double dbl_rcp_fact9  = 2.75573192239858906526e-06;  /* 1/9! */
	static constexpr double dbl_rcp_fact10 = 2.75573192239858906526e-07;  /* 1/10! */
	static constexpr double dbl_rcp_fact11 = 2.50521083854417187751e-08;  /* 1/11! */
	static constexpr double dbl_rcp_fact12 = 2.08767569878680989792e-09;  /* 1/12! */
	static constexpr double dbl_rcp_fact13 = 1.60590438368216145994e-10;  /* 1/13! */

	static constexpr ALIGN(32) signed int  f4_base[4]      = { flt_base,   flt_base,   flt_base,   flt_base   };
	static constexpr ALIGN(32) float       f4_log2e[4]     = { flt_log2e,  flt_log2e,  flt_log2e,  flt_log2e  };
	static constexpr ALIGN(32) float       f4_ln2f_hi[4]   = { flt_ln2_hi, flt_ln2_hi, flt_ln2_hi, flt_ln2_hi };
	static constexpr ALIGN(32) float       f4_ln2f_lo[4]   = { flt_ln2_lo, flt_ln2_lo, flt_ln2_lo, flt_ln2_lo };
	static constexpr ALIGN(32) float       f4_rcp_fact0[4] = { flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0 };
	static constexpr ALIGN(32) float       f4_rcp_fact1[4] = { flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1 };
	static constexpr ALIGN(32) float       f4_rcp_fact2[4] = { flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2 };
	static constexpr ALIGN(32) float       f4_rcp_fact3[4] = { flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3 };
	static constexpr ALIGN(32) float       f4_rcp_fact4[4] = { flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4 };
	static constexpr ALIGN(32) float       f4_rcp_fact5[4] = { flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5 };
	static constexpr ALIGN(32) float       f4_rcp_fact6[4] = { flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6 };
	static constexpr ALIGN(32) float       f4_rcp_fact7[4] = { flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7 };

	static constexpr ALIGN(64) signed int  f8_base[8]      = { flt_base,   flt_base,   flt_base,   flt_base,   flt_base,   flt_base,   flt_base,   flt_base   };
	static constexpr ALIGN(64) float       f8_log2e[8]     = { flt_log2e,  flt_log2e,  flt_log2e,  flt_log2e,  flt_log2e,  flt_log2e,  flt_log2e,  flt_log2e  };
	static constexpr ALIGN(64) float       f8_ln2f_hi[8]   = { flt_ln2_hi, flt_ln2_hi, flt_ln2_hi, flt_ln2_hi, flt_ln2_hi, flt_ln2_hi, flt_ln2_hi, flt_ln2_hi };
	static constexpr ALIGN(64) float       f8_ln2f_lo[8]   = { flt_ln2_lo, flt_ln2_lo, flt_ln2_lo, flt_ln2_lo, flt_ln2_lo, flt_ln2_lo, flt_ln2_lo, flt_ln2_lo };
	static constexpr ALIGN(64) float       f8_rcp_fact0[8] = { flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0, flt_rcp_fact0 };
	static constexpr ALIGN(64) float       f8_rcp_fact1[8] = { flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1, flt_rcp_fact1 };
	static constexpr ALIGN(64) float       f8_rcp_fact2[8] = { flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2, flt_rcp_fact2 };
	static constexpr ALIGN(64) float       f8_rcp_fact3[8] = { flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3, flt_rcp_fact3 };
	static constexpr ALIGN(64) float       f8_rcp_fact4[8] = { flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4, flt_rcp_fact4 };
	static constexpr ALIGN(64) float       f8_rcp_fact5[8] = { flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5, flt_rcp_fact5 };
	static constexpr ALIGN(64) float       f8_rcp_fact6[8] = { flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6, flt_rcp_fact6 };
	static constexpr ALIGN(64) float       f8_rcp_fact7[8] = { flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7, flt_rcp_fact7 };

	// Function template exp

	extern float exp(float x)
	{
		// t = x * log2(e);
		float t = x * flt_log2e;
		// the sign of t
		signed int sign = *reinterpret_cast<signed int*>(&t) & 0x80000000;
		// the absolute value of t
		signed int value = *reinterpret_cast<signed int*>(&t) & 0x7fffffff;
		// keep the integer part
		signed int integer = static_cast<signed int>(*reinterpret_cast<float*>(&value));
		// keep the decimal part
		float decimal = *reinterpret_cast<float*>(&value) - static_cast<float>(integer);
		// the round number of t
		if (decimal >= flt_half)
			integer += sign ? -1 : 1;
		float round = static_cast<float>(integer);
		// x -= round * ln2;
		x -= round * flt_ln2_hi;
		x -= round * flt_ln2_lo;
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		float y = flt_rcp_fact7;
		y = y * x + flt_rcp_fact6;
		y = y * x + flt_rcp_fact5;
		y = y * x + flt_rcp_fact4;
		y = y * x + flt_rcp_fact3;
		y = y * x + flt_rcp_fact2;
		y = y * x + flt_rcp_fact1;
		y = y * x + flt_rcp_fact0;
		// y *= (float) 2^integer;
		integer = (integer + 127) << 23;
		y *= *reinterpret_cast<float*>(&integer);
		return y;
	}

	template<cpu_inst_type inst>
	__m128 expf4(__m128 xmm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m128 expf4<cpu_sse41>(__m128 xmm_x)
	{
		// t = x * log2(e);
		__m128 xmm_t = _mm_mul_ps(xmm_x, *reinterpret_cast<const __m128*>(f4_log2e));
		// r = round(t);
		__m128 xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(f4_ln2f_hi)));
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(f4_ln2f_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m128 xmm_y = *reinterpret_cast<const __m128*>(f4_rcp_fact7);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(f4_rcp_fact6));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(f4_rcp_fact5));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(f4_rcp_fact4));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(f4_rcp_fact3));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(f4_rcp_fact2));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(f4_rcp_fact1));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(f4_rcp_fact0));
		// i = 2^r;
		__m128i xmm_i = _mm_cvttps_epi32(xmm_r);
		xmm_i = _mm_add_epi32(xmm_i, *reinterpret_cast<const __m128i*>(f4_base));
		xmm_i = _mm_slli_epi32(xmm_i, 23);
		// y *= (float) i;
		xmm_y = _mm_mul_ps(xmm_y, _mm_castsi128_ps(xmm_i));
		return xmm_y;
	}

	template<>
	__m128 expf4<cpu_sse41 | cpu_fma>(__m128 xmm_x)
	{
		// t = x * log2(e);
		__m128 xmm_t = _mm_mul_ps(xmm_x, *reinterpret_cast<const __m128*>(f4_log2e));
		// r = round(t);
		__m128 xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(f4_ln2f_hi)));
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(f4_ln2f_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m128 xmm_y = *reinterpret_cast<const __m128*>(f4_rcp_fact7);
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(f4_rcp_fact6));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(f4_rcp_fact5));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(f4_rcp_fact4));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(f4_rcp_fact3));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(f4_rcp_fact2));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(f4_rcp_fact1));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(f4_rcp_fact0));
		// i = 2^r;
		__m128i xmm_i = _mm_cvttps_epi32(xmm_r);
		xmm_i = _mm_add_epi32(xmm_i, *reinterpret_cast<const __m128i*>(f4_base));
		xmm_i = _mm_slli_epi32(xmm_i, 23);
		// y *= (float) i;
		xmm_y = _mm_mul_ps(xmm_y, _mm_castsi128_ps(xmm_i));
		return xmm_y;
	}

	extern __m256 expf8(__m256 ymm_x)
	{
		// t = x * log2(e);
		__m256 ymm_t = _mm256_mul_ps(ymm_x, *reinterpret_cast<const __m256*>(f8_log2e));
		// r = round(t);
		__m256 ymm_r = _mm256_round_ps(ymm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, *reinterpret_cast<const __m256*>(f8_ln2f_hi)));
		ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, *reinterpret_cast<const __m256*>(f8_ln2f_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m256 ymm_y = *reinterpret_cast<const __m256*>(f8_rcp_fact7);
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(f8_rcp_fact6));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(f8_rcp_fact5));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(f8_rcp_fact4));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(f8_rcp_fact3));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(f8_rcp_fact2));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(f8_rcp_fact1));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(f8_rcp_fact0));
		// i = 2^r;
		__m256i ymm_i = _mm256_cvttps_epi32(ymm_r);
		ymm_i = _mm256_add_epi32(ymm_i, *reinterpret_cast<const __m256i*>(f8_base));
		ymm_i = _mm256_slli_epi32(ymm_i, 23);
		// y *= (float) i;
		ymm_y = _mm256_mul_ps(ymm_y, _mm256_castsi256_ps(ymm_i));
		return ymm_y;
	}

} // namespace core

#endif
