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

#ifndef __CORE_CPU_KERNEL_CONVERT_INT32_H__
#define __CORE_CPU_KERNEL_CONVERT_INT32_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_convert_int32
	template<class T, cpu_inst_type inst>
	struct kernel_convert_int32
	{
		void operator()(size_t n, const signed int *a, T *b) const
		{
			constexpr size_t block = 8;

			while (n >= block)
			{
				b[0] = static_cast<T>(a[0]);
				b[1] = static_cast<T>(a[1]);
				b[2] = static_cast<T>(a[2]);
				b[3] = static_cast<T>(a[3]);
				b[4] = static_cast<T>(a[4]);
				b[5] = static_cast<T>(a[5]);
				b[6] = static_cast<T>(a[6]);
				b[7] = static_cast<T>(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<T>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<signed char, cpu_none>
	{
		void operator()(size_t n, const signed int *a, signed char *b) const
		{
			constexpr size_t block = 8;
			constexpr signed int min = static_cast<signed int>(int8_min);
			constexpr signed int max = static_cast<signed int>(int8_max);

			while (n >= block)
			{
				b[0] = a[0] < min ? int8_min : a[0] > max ? int8_max : static_cast<signed char>(a[0]);
				b[1] = a[1] < min ? int8_min : a[1] > max ? int8_max : static_cast<signed char>(a[1]);
				b[2] = a[2] < min ? int8_min : a[2] > max ? int8_max : static_cast<signed char>(a[2]);
				b[3] = a[3] < min ? int8_min : a[3] > max ? int8_max : static_cast<signed char>(a[3]);
				b[4] = a[4] < min ? int8_min : a[4] > max ? int8_max : static_cast<signed char>(a[4]);
				b[5] = a[5] < min ? int8_min : a[5] > max ? int8_max : static_cast<signed char>(a[5]);
				b[6] = a[6] < min ? int8_min : a[6] > max ? int8_max : static_cast<signed char>(a[6]);
				b[7] = a[7] < min ? int8_min : a[7] > max ? int8_max : static_cast<signed char>(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? int8_min : a[i] > max ? int8_max : static_cast<signed char>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<unsigned char, cpu_none>
	{
		void operator()(size_t n, const signed int *a, unsigned char *b) const
		{
			constexpr size_t block = 8;
			constexpr signed int min = static_cast<signed int>(uint8_min);
			constexpr signed int max = static_cast<signed int>(uint8_max);

			while (n >= block)
			{
				b[0] = a[0] < min ? uint8_min : a[0] > max ? uint8_max : static_cast<unsigned char>(a[0]);
				b[1] = a[1] < min ? uint8_min : a[1] > max ? uint8_max : static_cast<unsigned char>(a[1]);
				b[2] = a[2] < min ? uint8_min : a[2] > max ? uint8_max : static_cast<unsigned char>(a[2]);
				b[3] = a[3] < min ? uint8_min : a[3] > max ? uint8_max : static_cast<unsigned char>(a[3]);
				b[4] = a[4] < min ? uint8_min : a[4] > max ? uint8_max : static_cast<unsigned char>(a[4]);
				b[5] = a[5] < min ? uint8_min : a[5] > max ? uint8_max : static_cast<unsigned char>(a[5]);
				b[6] = a[6] < min ? uint8_min : a[6] > max ? uint8_max : static_cast<unsigned char>(a[6]);
				b[7] = a[7] < min ? uint8_min : a[7] > max ? uint8_max : static_cast<unsigned char>(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? uint8_min : a[i] > max ? uint8_max : static_cast<unsigned char>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<signed short, cpu_none>
	{
		void operator()(size_t n, const signed int *a, signed short *b) const
		{
			constexpr size_t block = 8;
			constexpr signed int min = static_cast<signed int>(int16_min);
			constexpr signed int max = static_cast<signed int>(int16_max);

			while (n >= block)
			{
				b[0] = a[0] < min ? int16_min : a[0] > max ? int16_max : static_cast<signed short>(a[0]);
				b[1] = a[1] < min ? int16_min : a[1] > max ? int16_max : static_cast<signed short>(a[1]);
				b[2] = a[2] < min ? int16_min : a[2] > max ? int16_max : static_cast<signed short>(a[2]);
				b[3] = a[3] < min ? int16_min : a[3] > max ? int16_max : static_cast<signed short>(a[3]);
				b[4] = a[4] < min ? int16_min : a[4] > max ? int16_max : static_cast<signed short>(a[4]);
				b[5] = a[5] < min ? int16_min : a[5] > max ? int16_max : static_cast<signed short>(a[5]);
				b[6] = a[6] < min ? int16_min : a[6] > max ? int16_max : static_cast<signed short>(a[6]);
				b[7] = a[7] < min ? int16_min : a[7] > max ? int16_max : static_cast<signed short>(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? int16_min : a[i] > max ? int16_max : static_cast<signed short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<unsigned short, cpu_none>
	{
		void operator()(size_t n, const signed int *a, unsigned short *b) const
		{
			constexpr size_t block = 8;
			constexpr signed int min = static_cast<signed int>(uint16_min);
			constexpr signed int max = static_cast<signed int>(uint16_max);

			while (n >= block)
			{
				b[0] = a[0] < min ? uint16_min : a[0] > max ? uint16_max : static_cast<unsigned short>(a[0]);
				b[1] = a[1] < min ? uint16_min : a[1] > max ? uint16_max : static_cast<unsigned short>(a[1]);
				b[2] = a[2] < min ? uint16_min : a[2] > max ? uint16_max : static_cast<unsigned short>(a[2]);
				b[3] = a[3] < min ? uint16_min : a[3] > max ? uint16_max : static_cast<unsigned short>(a[3]);
				b[4] = a[4] < min ? uint16_min : a[4] > max ? uint16_max : static_cast<unsigned short>(a[4]);
				b[5] = a[5] < min ? uint16_min : a[5] > max ? uint16_max : static_cast<unsigned short>(a[5]);
				b[6] = a[6] < min ? uint16_min : a[6] > max ? uint16_max : static_cast<unsigned short>(a[6]);
				b[7] = a[7] < min ? uint16_min : a[7] > max ? uint16_max : static_cast<unsigned short>(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? uint16_min : a[i] > max ? uint16_max : static_cast<unsigned short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<signed char, cpu_sse41>
	{
		void operator()(size_t n, const signed int *a, signed char *b) const
		{
			constexpr size_t block = 16;
			constexpr signed int min = static_cast<signed int>(int8_min);
			constexpr signed int max = static_cast<signed int>(int8_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				xmm_b0 = _mm_packs_epi32(xmm_a0, xmm_a1);
				xmm_b1 = _mm_packs_epi32(xmm_a2, xmm_a3);
				xmm_b0 = _mm_packs_epi16(xmm_b0, xmm_b1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? int8_min : a[i] > max ? int8_max : static_cast<signed char>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<unsigned char, cpu_sse41>
	{
		void operator()(size_t n, const signed int *a, unsigned char *b) const
		{
			constexpr size_t block = 16;
			constexpr signed int min = static_cast<signed int>(uint8_min);
			constexpr signed int max = static_cast<signed int>(uint8_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				xmm_b0 = _mm_packus_epi32(xmm_a0, xmm_a1);
				xmm_b1 = _mm_packus_epi32(xmm_a2, xmm_a3);
				xmm_b0 = _mm_packus_epi16(xmm_b0, xmm_b1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? uint8_min : a[i] > max ? uint8_max : static_cast<unsigned char>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<signed short, cpu_sse41>
	{
		void operator()(size_t n, const signed int *a, signed short *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 8;
			constexpr signed int min = static_cast<signed int>(int16_min);
			constexpr signed int max = static_cast<signed int>(int16_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				xmm_b0 = _mm_packs_epi32(xmm_a0, xmm_a1);
				xmm_b1 = _mm_packs_epi32(xmm_a2, xmm_a3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				// data-type conversion
				xmm_b0 = _mm_packs_epi32(xmm_a0, xmm_a1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? int16_min : a[i] > max ? int16_max : static_cast<signed short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, const signed int *a, unsigned short *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 8;
			constexpr signed int min = static_cast<signed int>(uint16_min);
			constexpr signed int max = static_cast<signed int>(uint16_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				xmm_b0 = _mm_packus_epi32(xmm_a0, xmm_a1);
				xmm_b1 = _mm_packus_epi32(xmm_a2, xmm_a3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				// data-type conversion
				xmm_b0 = _mm_packus_epi32(xmm_a0, xmm_a1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? uint16_min : a[i] > max ? uint16_max : static_cast<unsigned short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<float, cpu_sse2>
	{
		void operator()(size_t n, const signed int *a, float *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				xmm_b1 = _mm_cvtepi32_ps(xmm_a1);
				xmm_b2 = _mm_cvtepi32_ps(xmm_a2);
				xmm_b3 = _mm_cvtepi32_ps(xmm_a3);
				// store data into memory
				_mm_storeu_ps(b, xmm_b0);
				_mm_storeu_ps(b + 4, xmm_b1);
				_mm_storeu_ps(b + 8, xmm_b2);
				_mm_storeu_ps(b + 12, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				// store data into memory
				_mm_storeu_ps(b, xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<float>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<double, cpu_sse2>
	{
		void operator()(size_t n, const signed int *a, double *b) const
		{
			constexpr size_t block = 4;
			__m128i xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi32_pd(xmm_a0);
				xmm_b1 = _mm_cvtepi32_pd(xmm_a1);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<double>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<signed char, cpu_avx2>
	{
		void operator()(size_t n, const signed int *a, signed char *b) const
		{
			constexpr size_t block = 32;
			constexpr signed int min = static_cast<signed int>(int8_min);
			constexpr signed int max = static_cast<signed int>(int8_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 4);
				xmm_a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 5);
				xmm_a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 6);
				xmm_a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 7);
				// data-type conversion
				ymm_b0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a0), xmm_a4, 1);
				ymm_b1 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a1), xmm_a5, 1);
				ymm_b2 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a2), xmm_a6, 1);
				ymm_b3 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a3), xmm_a7, 1);
				ymm_b0 = _mm256_packs_epi32(ymm_b0, ymm_b1);
				ymm_b1 = _mm256_packs_epi32(ymm_b2, ymm_b3);
				ymm_b0 = _mm256_packs_epi16(ymm_b0, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? int8_min : a[i] > max ? int8_max : static_cast<signed char>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, const signed int *a, unsigned char *b) const
		{
			constexpr size_t block = 32;
			constexpr signed int min = static_cast<signed int>(uint8_min);
			constexpr signed int max = static_cast<signed int>(uint8_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 4);
				xmm_a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 5);
				xmm_a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 6);
				xmm_a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 7);
				// data-type conversion
				ymm_b0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a0), xmm_a4, 1);
				ymm_b1 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a1), xmm_a5, 1);
				ymm_b2 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a2), xmm_a6, 1);
				ymm_b3 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a3), xmm_a7, 1);
				ymm_b0 = _mm256_packus_epi32(ymm_b0, ymm_b1);
				ymm_b1 = _mm256_packus_epi32(ymm_b2, ymm_b3);
				ymm_b0 = _mm256_packus_epi16(ymm_b0, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? uint8_min : a[i] > max ? uint8_max : static_cast<unsigned char>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed int *a, signed short *b) const
		{
			constexpr size_t block = 16;
			constexpr signed int min = static_cast<signed int>(int16_min);
			constexpr signed int max = static_cast<signed int>(int16_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256i ymm_b0, ymm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				ymm_b0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a0), xmm_a2, 1);
				ymm_b1 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a1), xmm_a3, 1);
				ymm_b0 = _mm256_packs_epi32(ymm_b0, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? int16_min : a[i] > max ? int16_max : static_cast<signed short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const signed int *a, unsigned short *b) const
		{
			constexpr size_t block = 16;
			constexpr signed int min = static_cast<signed int>(uint16_min);
			constexpr signed int max = static_cast<signed int>(uint16_max);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256i ymm_b0, ymm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				ymm_b0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a0), xmm_a2, 1);
				ymm_b1 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_a1), xmm_a3, 1);
				ymm_b0 = _mm256_packus_epi32(ymm_b0, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] < min ? uint16_min : a[i] > max ? uint16_max : static_cast<unsigned short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<float, cpu_avx>
	{
		void operator()(size_t n, const signed int *a, float *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n >= block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// data-type conversion
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				ymm_b1 = _mm256_cvtepi32_ps(ymm_a1);
				ymm_b2 = _mm256_cvtepi32_ps(ymm_a2);
				ymm_b3 = _mm256_cvtepi32_ps(ymm_a3);
				// store data into memory
				_mm256_storeu_ps(b, ymm_b0);
				_mm256_storeu_ps(b + 8, ymm_b1);
				_mm256_storeu_ps(b + 16, ymm_b0);
				_mm256_storeu_ps(b + 24, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				// store data into memory
				_mm256_storeu_ps(b, ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<float>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int32<double, cpu_avx>
	{
		void operator()(size_t n, const signed int *a, double *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				ymm_b0 = _mm256_cvtepi32_pd(xmm_a0);
				ymm_b1 = _mm256_cvtepi32_pd(xmm_a1);
				ymm_b2 = _mm256_cvtepi32_pd(xmm_a2);
				ymm_b3 = _mm256_cvtepi32_pd(xmm_a3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b0);
				_mm256_storeu_pd(b + 12, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi32_pd(xmm_a0);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<double>(a[i]);
		}
	};

} // namespace core

#endif
