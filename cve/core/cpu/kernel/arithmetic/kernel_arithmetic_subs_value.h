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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_SUBS_VALUE_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_SUBS_VALUE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_subs_value

	template<class T, cpu_inst_type inst>
	struct kernel_subs_value
	{
		void operator()(size_t n, T a, const T *b, T *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a - b[0];
				c[1] = a - b[1];
				c[2] = a - b[2];
				c[3] = a - b[3];
				c[4] = a - b[4];
				c[5] = a - b[5];
				c[6] = a - b[6];
				c[7] = a - b[7];
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a - b[i];
		}
	};

	template<>
	struct kernel_subs_value<signed char, cpu_none>
	{
		void operator()(size_t n, signed char a, const signed char *b, signed char *c) const
		{
			constexpr size_t block = 8;
			const signed char t = a >> 7 ? int8_min : int8_max;
			signed char c0, c1, c2, c3, c4, c5, c6, c7;
			signed char s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a - b[0];
				c1 = a - b[1];
				c2 = a - b[2];
				c3 = a - b[3];
				c4 = a - b[4];
				c5 = a - b[5];
				c6 = a - b[6];
				c7 = a - b[7];
				s0 = (a ^ b[0]) & (c0 ^ a);
				s1 = (a ^ b[1]) & (c1 ^ a);
				s2 = (a ^ b[2]) & (c2 ^ a);
				s3 = (a ^ b[3]) & (c3 ^ a);
				s4 = (a ^ b[4]) & (c4 ^ a);
				s5 = (a ^ b[5]) & (c5 ^ a);
				s6 = (a ^ b[6]) & (c6 ^ a);
				s7 = (a ^ b[7]) & (c7 ^ a);
				c[0] = s0 >> 7 ? t : c0;
				c[1] = s1 >> 7 ? t : c1;
				c[2] = s2 >> 7 ? t : c2;
				c[3] = s3 >> 7 ? t : c3;
				c[4] = s4 >> 7 ? t : c4;
				c[5] = s5 >> 7 ? t : c5;
				c[6] = s6 >> 7 ? t : c6;
				c[7] = s7 >> 7 ? t : c7;
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a - b[i];
				s0 = (a ^ b[i]) & (c0 ^ a);
				c[i] = s0 >> 7 ? t : c0;
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned char, cpu_none>
	{
		void operator()(size_t n, unsigned char a, const unsigned char *b, unsigned char *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a < b[0] ? uint8_min : a - b[0];
				c[1] = a < b[1] ? uint8_min : a - b[1];
				c[2] = a < b[2] ? uint8_min : a - b[2];
				c[3] = a < b[3] ? uint8_min : a - b[3];
				c[4] = a < b[4] ? uint8_min : a - b[4];
				c[5] = a < b[5] ? uint8_min : a - b[5];
				c[6] = a < b[6] ? uint8_min : a - b[6];
				c[7] = a < b[7] ? uint8_min : a - b[7];
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a < b[i] ? uint8_min : a - b[i];
		}
	};

	template<>
	struct kernel_subs_value<signed short, cpu_none>
	{
		void operator()(size_t n, signed short a, const signed short *b, signed short *c) const
		{
			constexpr size_t block = 8;
			const signed short t = a >> 15 ? int16_min : int16_max;
			signed short c0, c1, c2, c3, c4, c5, c6, c7;
			signed short s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a - b[0];
				c1 = a - b[1];
				c2 = a - b[2];
				c3 = a - b[3];
				c4 = a - b[4];
				c5 = a - b[5];
				c6 = a - b[6];
				c7 = a - b[7];
				s0 = (a ^ b[0]) & (c0 ^ a);
				s1 = (a ^ b[1]) & (c1 ^ a);
				s2 = (a ^ b[2]) & (c2 ^ a);
				s3 = (a ^ b[3]) & (c3 ^ a);
				s4 = (a ^ b[4]) & (c4 ^ a);
				s5 = (a ^ b[5]) & (c5 ^ a);
				s6 = (a ^ b[6]) & (c6 ^ a);
				s7 = (a ^ b[7]) & (c7 ^ a);
				c[0] = s0 >> 15 ? t : c0;
				c[1] = s1 >> 15 ? t : c1;
				c[2] = s2 >> 15 ? t : c2;
				c[3] = s3 >> 15 ? t : c3;
				c[4] = s4 >> 15 ? t : c4;
				c[5] = s5 >> 15 ? t : c5;
				c[6] = s6 >> 15 ? t : c6;
				c[7] = s7 >> 15 ? t : c7;
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a - b[i];
				s0 = (a ^ b[i]) & (c0 ^ a);
				c[i] = s0 >> 15 ? t : c0;
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned short, cpu_none>
	{
		void operator()(size_t n, unsigned short a, const unsigned short *b, unsigned short *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a < b[0] ? uint16_min : a - b[0];
				c[1] = a < b[1] ? uint16_min : a - b[1];
				c[2] = a < b[2] ? uint16_min : a - b[2];
				c[3] = a < b[3] ? uint16_min : a - b[3];
				c[4] = a < b[4] ? uint16_min : a - b[4];
				c[5] = a < b[5] ? uint16_min : a - b[5];
				c[6] = a < b[6] ? uint16_min : a - b[6];
				c[7] = a < b[7] ? uint16_min : a - b[7];
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a < b[i] ? uint16_min : a - b[i];
		}
	};

	template<>
	struct kernel_subs_value<signed int, cpu_none>
	{
		void operator()(size_t n, signed int a, const signed int *b, signed int *c) const
		{
			constexpr size_t block = 8;
			const signed int t = a >> 31 ? int32_min : int32_max;
			signed int c0, c1, c2, c3, c4, c5, c6, c7;
			signed int s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a - b[0];
				c1 = a - b[1];
				c2 = a - b[2];
				c3 = a - b[3];
				c4 = a - b[4];
				c5 = a - b[5];
				c6 = a - b[6];
				c7 = a - b[7];
				s0 = (a ^ b[0]) & (c0 ^ a);
				s1 = (a ^ b[1]) & (c1 ^ a);
				s2 = (a ^ b[2]) & (c2 ^ a);
				s3 = (a ^ b[3]) & (c3 ^ a);
				s4 = (a ^ b[4]) & (c4 ^ a);
				s5 = (a ^ b[5]) & (c5 ^ a);
				s6 = (a ^ b[6]) & (c6 ^ a);
				s7 = (a ^ b[7]) & (c7 ^ a);
				c[0] = s0 >> 31 ? t : c0;
				c[1] = s1 >> 31 ? t : c1;
				c[2] = s2 >> 31 ? t : c2;
				c[3] = s3 >> 31 ? t : c3;
				c[4] = s4 >> 31 ? t : c4;
				c[5] = s5 >> 31 ? t : c5;
				c[6] = s6 >> 31 ? t : c6;
				c[7] = s7 >> 31 ? t : c7;
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a - b[i];
				s0 = (a ^ b[i]) & (c0 ^ a);
				c[i] = s0 >> 31 ? t : c0;
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned int, cpu_none>
	{
		void operator()(size_t n, unsigned int a, const unsigned int *b, unsigned int *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a < b[0] ? uint32_min : a - b[0];
				c[1] = a < b[1] ? uint32_min : a - b[1];
				c[2] = a < b[2] ? uint32_min : a - b[2];
				c[3] = a < b[3] ? uint32_min : a - b[3];
				c[4] = a < b[4] ? uint32_min : a - b[4];
				c[5] = a < b[5] ? uint32_min : a - b[5];
				c[6] = a < b[6] ? uint32_min : a - b[6];
				c[7] = a < b[7] ? uint32_min : a - b[7];
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a < b[i] ? uint32_min : a - b[i];
		}
	};

	template<>
	struct kernel_subs_value<signed char, cpu_sse2>
	{
		void operator()(size_t n, signed char a, const signed char *b, signed char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_a = _mm_set1_epi8(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epi8(xmm_a, xmm_b0);
				xmm_c1 = _mm_subs_epi8(xmm_a, xmm_b1);
				xmm_c2 = _mm_subs_epi8(xmm_a, xmm_b2);
				xmm_c3 = _mm_subs_epi8(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epi8(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_b0 = _mm_set1_epi8(b[i]);
				xmm_c0 = _mm_subs_epi8(xmm_a, xmm_b0);
				c[i] = reinterpret_cast<signed char*>(&xmm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned char, cpu_sse2>
	{
		void operator()(size_t n, unsigned char a, const unsigned char *b, unsigned char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_a = _mm_set1_epi8(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epu8(xmm_a, xmm_b0);
				xmm_c1 = _mm_subs_epu8(xmm_a, xmm_b1);
				xmm_c2 = _mm_subs_epu8(xmm_a, xmm_b2);
				xmm_c3 = _mm_subs_epu8(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epu8(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_b0 = _mm_set1_epi8(b[i]);
				xmm_c0 = _mm_subs_epu8(xmm_a, xmm_b0);
				c[i] = reinterpret_cast<unsigned char*>(&xmm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<signed short, cpu_sse2>
	{
		void operator()(size_t n, signed short a, const signed short *b, signed short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_a = _mm_set1_epi16(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epi16(xmm_a, xmm_b0);
				xmm_c1 = _mm_subs_epi16(xmm_a, xmm_b1);
				xmm_c2 = _mm_subs_epi16(xmm_a, xmm_b2);
				xmm_c3 = _mm_subs_epi16(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epi16(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_b0 = _mm_set1_epi16(b[i]);
				xmm_c0 = _mm_subs_epi16(xmm_a, xmm_b0);
				c[i] = reinterpret_cast<signed short*>(&xmm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned short, cpu_sse2>
	{
		void operator()(size_t n, unsigned short a, const unsigned short *b, unsigned short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_a = _mm_set1_epi16(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epu16(xmm_a, xmm_b0);
				xmm_c1 = _mm_subs_epu16(xmm_a, xmm_b1);
				xmm_c2 = _mm_subs_epu16(xmm_a, xmm_b2);
				xmm_c3 = _mm_subs_epu16(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = subs(a, b);
				xmm_c0 = _mm_subs_epu16(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_b0 = _mm_set1_epi16(b[i]);
				xmm_c0 = _mm_subs_epu16(xmm_a, xmm_b0);
				c[i] = reinterpret_cast<unsigned short*>(&xmm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<signed int, cpu_sse2>
	{
		void operator()(size_t n, signed int a, const signed int *b, signed int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			signed int r, s;
			signed int t = a >> 31 ? int32_min : int32_max;
			const __m128i xmm_t = _mm_set1_epi32(t);
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a - b;
				xmm_c0 = _mm_sub_epi32(xmm_a, xmm_b0);
				xmm_c1 = _mm_sub_epi32(xmm_a, xmm_b1);
				xmm_c2 = _mm_sub_epi32(xmm_a, xmm_b2);
				xmm_c3 = _mm_sub_epi32(xmm_a, xmm_b3);
				// s = (a ^ b) & (c ^ a);
				xmm_s0 = _mm_and_si128(_mm_or_si128(xmm_a, xmm_b0), _mm_or_si128(xmm_c0, xmm_a));
				xmm_s1 = _mm_and_si128(_mm_or_si128(xmm_a, xmm_b1), _mm_or_si128(xmm_c1, xmm_a));
				xmm_s2 = _mm_and_si128(_mm_or_si128(xmm_a, xmm_b2), _mm_or_si128(xmm_c2, xmm_a));
				xmm_s3 = _mm_and_si128(_mm_or_si128(xmm_a, xmm_b3), _mm_or_si128(xmm_c3, xmm_a));
				// c = s < 0 ? t : c;
				xmm_s0 = _mm_srai_epi32(xmm_s0, 31);
				xmm_s1 = _mm_srai_epi32(xmm_s1, 31);
				xmm_s2 = _mm_srai_epi32(xmm_s2, 31);
				xmm_s3 = _mm_srai_epi32(xmm_s3, 31);
				xmm_c0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_t), _mm_andnot_si128(xmm_s0, xmm_c0));
				xmm_c1 = _mm_or_si128(_mm_and_si128(xmm_s1, xmm_t), _mm_andnot_si128(xmm_s1, xmm_c1));
				xmm_c2 = _mm_or_si128(_mm_and_si128(xmm_s2, xmm_t), _mm_andnot_si128(xmm_s2, xmm_c2));
				xmm_c3 = _mm_or_si128(_mm_and_si128(xmm_s3, xmm_t), _mm_andnot_si128(xmm_s3, xmm_c3));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a - b;
				xmm_c0 = _mm_sub_epi32(xmm_a, xmm_b0);
				// s = (a ^ b) & (c ^ a);
				xmm_s0 = _mm_and_si128(_mm_or_si128(xmm_a, xmm_b0), _mm_or_si128(xmm_c0, xmm_a));
				// c = s < 0 ? t : c;
				xmm_s0 = _mm_srai_epi32(xmm_s0, 31);
				xmm_c0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_t), _mm_andnot_si128(xmm_s0, xmm_c0));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				r = a - b[i];
				s = (a ^ b[i]) & (r ^ a);
				c[i] = s >> 31 ? t : r;
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, unsigned int a, const unsigned int *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_min = _mm_set1_epi32(uint32_min);
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a - b;
				xmm_c0 = _mm_sub_epi32(xmm_a, xmm_b0);
				xmm_c1 = _mm_sub_epi32(xmm_a, xmm_b1);
				xmm_c2 = _mm_sub_epi32(xmm_a, xmm_b2);
				xmm_c3 = _mm_sub_epi32(xmm_a, xmm_b3);
				// s = b > a;
				xmm_s0 = _mm_srai_epi32(_mm_xor_si128(xmm_b0, xmm_a), 31);
				xmm_s1 = _mm_srai_epi32(_mm_xor_si128(xmm_b1, xmm_a), 31);
				xmm_s2 = _mm_srai_epi32(_mm_xor_si128(xmm_b2, xmm_a), 31);
				xmm_s3 = _mm_srai_epi32(_mm_xor_si128(xmm_b3, xmm_a), 31);
				xmm_s0 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_b0, xmm_a), xmm_s0);
				xmm_s1 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_b1, xmm_a), xmm_s1);
				xmm_s2 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_b2, xmm_a), xmm_s2);
				xmm_s3 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_b3, xmm_a), xmm_s3);
				// c = s ? min : c;
				xmm_c0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_min), _mm_andnot_si128(xmm_s0, xmm_c0));
				xmm_c1 = _mm_or_si128(_mm_and_si128(xmm_s1, xmm_min), _mm_andnot_si128(xmm_s1, xmm_c1));
				xmm_c2 = _mm_or_si128(_mm_and_si128(xmm_s2, xmm_min), _mm_andnot_si128(xmm_s2, xmm_c2));
				xmm_c3 = _mm_or_si128(_mm_and_si128(xmm_s3, xmm_min), _mm_andnot_si128(xmm_s3, xmm_c3));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a - b;
				xmm_c0 = _mm_sub_epi32(xmm_a, xmm_b0);
				// s = b > a;
				xmm_s0 = _mm_srai_epi32(_mm_xor_si128(xmm_b0, xmm_a), 31);
				xmm_s0 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_b0, xmm_a), xmm_s0);
				// c = s ? max : c;
				xmm_c0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_min), _mm_andnot_si128(xmm_s0, xmm_c0));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = b[i] > a ? uint32_min : a - b[i];
		}
	};

	template<>
	struct kernel_subs_value<signed char, cpu_avx2>
	{
		void operator()(size_t n, signed char a, const signed char *b, signed char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_a = _mm256_set1_epi8(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epi8(ymm_a, ymm_b0);
				ymm_c1 = _mm256_subs_epi8(ymm_a, ymm_b1);
				ymm_c2 = _mm256_subs_epi8(ymm_a, ymm_b2);
				ymm_c3 = _mm256_subs_epi8(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epi8(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_b0 = _mm256_set1_epi8(b[i]);
				ymm_c0 = _mm256_subs_epi8(ymm_a, ymm_b0);
				c[i] = reinterpret_cast<signed char*>(&ymm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, unsigned char a, const unsigned char *b, unsigned char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_a = _mm256_set1_epi8(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epu8(ymm_a, ymm_b0);
				ymm_c1 = _mm256_subs_epu8(ymm_a, ymm_b1);
				ymm_c2 = _mm256_subs_epu8(ymm_a, ymm_b2);
				ymm_c3 = _mm256_subs_epu8(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epu8(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_b0 = _mm256_set1_epi8(b[i]);
				ymm_c0 = _mm256_subs_epu8(ymm_a, ymm_b0);
				c[i] = reinterpret_cast<unsigned char*>(&ymm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<signed short, cpu_avx2>
	{
		void operator()(size_t n, signed short a, const signed short *b, signed short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_a = _mm256_set1_epi16(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epi16(ymm_a, ymm_b0);
				ymm_c1 = _mm256_subs_epi16(ymm_a, ymm_b1);
				ymm_c2 = _mm256_subs_epi16(ymm_a, ymm_b2);
				ymm_c3 = _mm256_subs_epi16(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epi16(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_b0 = _mm256_set1_epi16(b[i]);
				ymm_c0 = _mm256_subs_epi16(ymm_a, ymm_b0);
				c[i] = reinterpret_cast<signed short*>(&ymm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, unsigned short a, const unsigned short *b, unsigned short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_a = _mm256_set1_epi16(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epu16(ymm_a, ymm_b0);
				ymm_c1 = _mm256_subs_epu16(ymm_a, ymm_b1);
				ymm_c2 = _mm256_subs_epu16(ymm_a, ymm_b2);
				ymm_c3 = _mm256_subs_epu16(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = subs(a, b);
				ymm_c0 = _mm256_subs_epu16(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_b0 = _mm256_set1_epi16(b[i]);
				ymm_c0 = _mm256_subs_epu16(ymm_a, ymm_b0);
				c[i] = reinterpret_cast<unsigned short*>(&ymm_c0)[0];
			}
		}
	};

	template<>
	struct kernel_subs_value<signed int, cpu_avx2>
	{
		void operator()(size_t n, signed int a, const signed int *b, signed int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			signed int r, s;
			signed int t = a >> 31 ? int32_min : int32_max;
			const __m256i ymm_t = _mm256_set1_epi32(t);
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a - b;
				ymm_c0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_sub_epi32(ymm_a, ymm_b1);
				ymm_c2 = _mm256_sub_epi32(ymm_a, ymm_b2);
				ymm_c3 = _mm256_sub_epi32(ymm_a, ymm_b3);
				// s = (a ^ b) & (c ^ a);
				ymm_s0 = _mm256_and_si256(_mm256_or_si256(ymm_a, ymm_b0), _mm256_or_si256(ymm_c0, ymm_a));
				ymm_s1 = _mm256_and_si256(_mm256_or_si256(ymm_a, ymm_b1), _mm256_or_si256(ymm_c1, ymm_a));
				ymm_s2 = _mm256_and_si256(_mm256_or_si256(ymm_a, ymm_b2), _mm256_or_si256(ymm_c2, ymm_a));
				ymm_s3 = _mm256_and_si256(_mm256_or_si256(ymm_a, ymm_b3), _mm256_or_si256(ymm_c3, ymm_a));
				// c = s < 0 ? t : c;
				ymm_s0 = _mm256_srai_epi32(ymm_s0, 31);
				ymm_s1 = _mm256_srai_epi32(ymm_s1, 31);
				ymm_s2 = _mm256_srai_epi32(ymm_s2, 31);
				ymm_s3 = _mm256_srai_epi32(ymm_s3, 31);
				ymm_c0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_t), _mm256_andnot_si256(ymm_s0, ymm_c0));
				ymm_c1 = _mm256_or_si256(_mm256_and_si256(ymm_s1, ymm_t), _mm256_andnot_si256(ymm_s1, ymm_c1));
				ymm_c2 = _mm256_or_si256(_mm256_and_si256(ymm_s2, ymm_t), _mm256_andnot_si256(ymm_s2, ymm_c2));
				ymm_c3 = _mm256_or_si256(_mm256_and_si256(ymm_s3, ymm_t), _mm256_andnot_si256(ymm_s3, ymm_c3));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a - b;
				ymm_c0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				// s = (a ^ b) & (c ^ a);
				ymm_s0 = _mm256_and_si256(_mm256_or_si256(ymm_a, ymm_b0), _mm256_or_si256(ymm_c0, ymm_a));
				// c = s < 0 ? t : c;
				ymm_s0 = _mm256_srai_epi32(ymm_s0, 31);
				ymm_c0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_t), _mm256_andnot_si256(ymm_s0, ymm_c0));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				r = a - b[i];
				s = (a ^ b[i]) & (r ^ a);
				c[i] = s >> 31 ? t : r;
			}
		}
	};

	template<>
	struct kernel_subs_value<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const unsigned int *b, unsigned int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_min = _mm256_set1_epi32(uint32_min);
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a - b;
				ymm_c0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_sub_epi32(ymm_a, ymm_b1);
				ymm_c2 = _mm256_sub_epi32(ymm_a, ymm_b2);
				ymm_c3 = _mm256_sub_epi32(ymm_a, ymm_b3);
				// s = b > a;
				ymm_s0 = _mm256_srai_epi32(_mm256_xor_si256(ymm_b0, ymm_a), 31);
				ymm_s1 = _mm256_srai_epi32(_mm256_xor_si256(ymm_b1, ymm_a), 31);
				ymm_s2 = _mm256_srai_epi32(_mm256_xor_si256(ymm_b2, ymm_a), 31);
				ymm_s3 = _mm256_srai_epi32(_mm256_xor_si256(ymm_b3, ymm_a), 31);
				ymm_s0 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_b0, ymm_a), ymm_s0);
				ymm_s1 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_b1, ymm_a), ymm_s1);
				ymm_s2 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_b2, ymm_a), ymm_s2);
				ymm_s3 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_b3, ymm_a), ymm_s3);
				// c = s ? min : c;
				ymm_c0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_min), _mm256_andnot_si256(ymm_s0, ymm_c0));
				ymm_c1 = _mm256_or_si256(_mm256_and_si256(ymm_s1, ymm_min), _mm256_andnot_si256(ymm_s1, ymm_c1));
				ymm_c2 = _mm256_or_si256(_mm256_and_si256(ymm_s2, ymm_min), _mm256_andnot_si256(ymm_s2, ymm_c2));
				ymm_c3 = _mm256_or_si256(_mm256_and_si256(ymm_s3, ymm_min), _mm256_andnot_si256(ymm_s3, ymm_c3));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a - b;
				ymm_c0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				// s = a > c;
				ymm_s0 = _mm256_srai_epi32(_mm256_xor_si256(ymm_a, ymm_c0), 31);
				ymm_s0 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a, ymm_c0), ymm_s0);
				// c = s ? min : c;
				ymm_c0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_min), _mm256_andnot_si256(ymm_s0, ymm_c0));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = b[i] > a ? uint32_min : a - b[i];
		}
	};

} // namespace core

#endif
