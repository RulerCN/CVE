/*====================================================================
Copyright (C) 2016-2016 Ruler. All rights reserved.
Author:  Ruler
Address: Nan'an District,Chongqing,China
Contact: 26105499@qq.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. The name of the author may not be used to endorse or promote
products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY RULER ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
====================================================================*/
#pragma once

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_ADDS_VALUE_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_ADDS_VALUE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_adds_value

	template<class T, cpu_inst_type inst>
	struct kernel_adds_value
	{
		void operator()(size_t n, const T *a, const T b, T *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a[0] + b;
				c[1] = a[1] + b;
				c[2] = a[2] + b;
				c[3] = a[3] + b;
				c[4] = a[4] + b;
				c[5] = a[5] + b;
				c[6] = a[6] + b;
				c[7] = a[7] + b;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<signed char, cpu_none>
	{
		void operator()(size_t n, const signed char *a, const signed char b, signed char *c) const
		{
			constexpr size_t block = 8;
			const signed char d = b >> 7 ? int8_min : int8_max;
			signed char c0, c1, c2, c3, c4, c5, c6, c7;
			signed char s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a[0] + b;
				c1 = a[1] + b;
				c2 = a[2] + b;
				c3 = a[3] + b;
				c4 = a[4] + b;
				c5 = a[5] + b;
				c6 = a[6] + b;
				c7 = a[7] + b;
				s0 = ~(a[0] ^ b) & (c0 ^ a[0]);
				s1 = ~(a[1] ^ b) & (c1 ^ a[1]);
				s2 = ~(a[2] ^ b) & (c2 ^ a[2]);
				s3 = ~(a[3] ^ b) & (c3 ^ a[3]);
				s4 = ~(a[4] ^ b) & (c4 ^ a[4]);
				s5 = ~(a[5] ^ b) & (c5 ^ a[5]);
				s6 = ~(a[6] ^ b) & (c6 ^ a[6]);
				s7 = ~(a[7] ^ b) & (c7 ^ a[7]);
				c[0] = s0 >> 7 ? d : c0;
				c[1] = s1 >> 7 ? d : c1;
				c[2] = s2 >> 7 ? d : c2;
				c[3] = s3 >> 7 ? d : c3;
				c[4] = s4 >> 7 ? d : c4;
				c[5] = s5 >> 7 ? d : c5;
				c[6] = s6 >> 7 ? d : c6;
				c[7] = s7 >> 7 ? d : c7;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b;
				s0 = ~(a[i] ^ b) & (c0 ^ a[i]);
				c[i] = s0 >> 7 ? d : c0;
			}
		}
	};

	template<>
	struct kernel_adds_value<unsigned char, cpu_none>
	{
		void operator()(size_t n, const unsigned char *a, const unsigned char b, unsigned char *c) const
		{
			constexpr size_t block = 8;
			unsigned char c0, c1, c2, c3, c4, c5, c6, c7;

			while (n > block)
			{
				c0 = a[0] + b;
				c1 = a[1] + b;
				c2 = a[2] + b;
				c3 = a[3] + b;
				c4 = a[4] + b;
				c5 = a[5] + b;
				c6 = a[6] + b;
				c7 = a[7] + b;
				c[0] = c0 < b ? uint8_max : c0;
				c[1] = c1 < b ? uint8_max : c1;
				c[2] = c2 < b ? uint8_max : c2;
				c[3] = c3 < b ? uint8_max : c3;
				c[4] = c4 < b ? uint8_max : c4;
				c[5] = c5 < b ? uint8_max : c5;
				c[6] = c6 < b ? uint8_max : c6;
				c[7] = c7 < b ? uint8_max : c7;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b;
				c[i] = c0 < b ? uint8_max : c0;
			}
		}
	};

	template<>
	struct kernel_adds_value<signed short, cpu_none>
	{
		void operator()(size_t n, const signed short *a, const signed short b, signed short *c) const
		{
			constexpr size_t block = 8;
			const signed short d = b >> 15 ? int16_min : int16_max;
			signed short c0, c1, c2, c3, c4, c5, c6, c7;
			signed short s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a[0] + b;
				c1 = a[1] + b;
				c2 = a[2] + b;
				c3 = a[3] + b;
				c4 = a[4] + b;
				c5 = a[5] + b;
				c6 = a[6] + b;
				c7 = a[7] + b;
				s0 = ~(a[0] ^ b) & (c0 ^ a[0]);
				s1 = ~(a[1] ^ b) & (c1 ^ a[1]);
				s2 = ~(a[2] ^ b) & (c2 ^ a[2]);
				s3 = ~(a[3] ^ b) & (c3 ^ a[3]);
				s4 = ~(a[4] ^ b) & (c4 ^ a[4]);
				s5 = ~(a[5] ^ b) & (c5 ^ a[5]);
				s6 = ~(a[6] ^ b) & (c6 ^ a[6]);
				s7 = ~(a[7] ^ b) & (c7 ^ a[7]);
				c[0] = s0 >> 15 ? d : c0;
				c[1] = s1 >> 15 ? d : c1;
				c[2] = s2 >> 15 ? d : c2;
				c[3] = s3 >> 15 ? d : c3;
				c[4] = s4 >> 15 ? d : c4;
				c[5] = s5 >> 15 ? d : c5;
				c[6] = s6 >> 15 ? d : c6;
				c[7] = s7 >> 15 ? d : c7;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b;
				s0 = ~(a[i] ^ b) & (c0 ^ a[i]);
				c[i] = s0 >> 15 ? d : c0;
			}
		}
	};

	template<>
	struct kernel_adds_value<unsigned short, cpu_none>
	{
		void operator()(size_t n, const unsigned short *a, const unsigned short b, unsigned short *c) const
		{
			constexpr size_t block = 8;
			unsigned short c0, c1, c2, c3, c4, c5, c6, c7;

			while (n > block)
			{
				c0 = a[0] + b;
				c1 = a[1] + b;
				c2 = a[2] + b;
				c3 = a[3] + b;
				c4 = a[4] + b;
				c5 = a[5] + b;
				c6 = a[6] + b;
				c7 = a[7] + b;
				c[0] = c0 < b ? uint16_max : c0;
				c[1] = c1 < b ? uint16_max : c1;
				c[2] = c2 < b ? uint16_max : c2;
				c[3] = c3 < b ? uint16_max : c3;
				c[4] = c4 < b ? uint16_max : c4;
				c[5] = c5 < b ? uint16_max : c5;
				c[6] = c6 < b ? uint16_max : c6;
				c[7] = c7 < b ? uint16_max : c7;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b;
				c[i] = c0 < b ? uint16_max : c0;
			}
		}
	};

	template<>
	struct kernel_adds_value<signed int, cpu_none>
	{
		void operator()(size_t n, const signed int *a, const signed int b, signed int *c) const
		{
			constexpr size_t block = 8;
			const signed int d = b >> 31 ? int32_min : int32_max;
			signed int c0, c1, c2, c3, c4, c5, c6, c7;
			signed int s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a[0] + b;
				c1 = a[1] + b;
				c2 = a[2] + b;
				c3 = a[3] + b;
				c4 = a[4] + b;
				c5 = a[5] + b;
				c6 = a[6] + b;
				c7 = a[7] + b;
				s0 = ~(a[0] ^ b) & (c0 ^ a[0]);
				s1 = ~(a[1] ^ b) & (c1 ^ a[1]);
				s2 = ~(a[2] ^ b) & (c2 ^ a[2]);
				s3 = ~(a[3] ^ b) & (c3 ^ a[3]);
				s4 = ~(a[4] ^ b) & (c4 ^ a[4]);
				s5 = ~(a[5] ^ b) & (c5 ^ a[5]);
				s6 = ~(a[6] ^ b) & (c6 ^ a[6]);
				s7 = ~(a[7] ^ b) & (c7 ^ a[7]);
				c[0] = s0 >> 31 ? d : c0;
				c[1] = s1 >> 31 ? d : c1;
				c[2] = s2 >> 31 ? d : c2;
				c[3] = s3 >> 31 ? d : c3;
				c[4] = s4 >> 31 ? d : c4;
				c[5] = s5 >> 31 ? d : c5;
				c[6] = s6 >> 31 ? d : c6;
				c[7] = s7 >> 31 ? d : c7;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b;
				s0 = ~(a[i] ^ b) & (c0 ^ a[i]);
				c[i] = s0 >> 31 ? d : c0;
			}
		}
	};

	template<>
	struct kernel_adds_value<unsigned int, cpu_none>
	{
		void operator()(size_t n, const unsigned int *a, const unsigned int b, unsigned int *c) const
		{
			constexpr size_t block = 8;
			unsigned int c0, c1, c2, c3, c4, c5, c6, c7;

			while (n > block)
			{
				c0 = a[0] + b;
				c1 = a[1] + b;
				c2 = a[2] + b;
				c3 = a[3] + b;
				c4 = a[4] + b;
				c5 = a[5] + b;
				c6 = a[6] + b;
				c7 = a[7] + b;
				c[0] = c0 < b ? uint32_max : c0;
				c[1] = c1 < b ? uint32_max : c1;
				c[2] = c2 < b ? uint32_max : c2;
				c[3] = c3 < b ? uint32_max : c3;
				c[4] = c4 < b ? uint32_max : c4;
				c[5] = c5 < b ? uint32_max : c5;
				c[6] = c6 < b ? uint32_max : c6;
				c[7] = c7 < b ? uint32_max : c7;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b;
				c[i] = c0 < b ? uint32_max : c0;
			}
		}
	};

	template<>
	struct kernel_adds_value<signed char, cpu_sse2>
	{
		void operator()(size_t n, const signed char *a, const signed char b, signed char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_b = _mm_set1_epi8(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi8(xmm_a0, xmm_b);
				xmm_c1 = _mm_adds_epi8(xmm_a1, xmm_b);
				xmm_c2 = _mm_adds_epi8(xmm_a2, xmm_b);
				xmm_c3 = _mm_adds_epi8(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi8(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<unsigned char, cpu_sse2>
	{
		void operator()(size_t n, const unsigned char *a, const unsigned char b, unsigned char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_b = _mm_set1_epi8(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi8(xmm_a0, xmm_b);
				xmm_c1 = _mm_adds_epi8(xmm_a1, xmm_b);
				xmm_c2 = _mm_adds_epi8(xmm_a2, xmm_b);
				xmm_c3 = _mm_adds_epi8(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi8(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<signed short, cpu_sse2>
	{
		void operator()(size_t n, const signed short *a, const signed short b, signed short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_b = _mm_set1_epi16(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi16(xmm_a0, xmm_b);
				xmm_c1 = _mm_adds_epi16(xmm_a1, xmm_b);
				xmm_c2 = _mm_adds_epi16(xmm_a2, xmm_b);
				xmm_c3 = _mm_adds_epi16(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi16(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<unsigned short, cpu_sse2>
	{
		void operator()(size_t n, const unsigned short *a, const unsigned short b, unsigned short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_b = _mm_set1_epi16(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi16(xmm_a0, xmm_b);
				xmm_c1 = _mm_adds_epi16(xmm_a1, xmm_b);
				xmm_c2 = _mm_adds_epi16(xmm_a2, xmm_b);
				xmm_c3 = _mm_adds_epi16(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = adds(a, b);
				xmm_c0 = _mm_adds_epi16(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<signed int, cpu_sse2>
	{
		void operator()(size_t n, const signed int *a, const signed int b, signed int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			signed int r, s;
			signed int t = b >> 31 ? int32_min : int32_max;
			const __m128i xmm_b = _mm_set1_epi32(b);
			const __m128i xmm_t = _mm_set1_epi32(t);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi32(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi32(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi32(xmm_a3, xmm_b);
				// s = ~(a ^ b) & (c ^ a);
				xmm_s0 = _mm_andnot_si128(_mm_or_si128(xmm_a0, xmm_b), _mm_or_si128(xmm_c0, xmm_a0));
				xmm_s1 = _mm_andnot_si128(_mm_or_si128(xmm_a1, xmm_b), _mm_or_si128(xmm_c1, xmm_a1));
				xmm_s2 = _mm_andnot_si128(_mm_or_si128(xmm_a2, xmm_b), _mm_or_si128(xmm_c2, xmm_a2));
				xmm_s3 = _mm_andnot_si128(_mm_or_si128(xmm_a3, xmm_b), _mm_or_si128(xmm_c3, xmm_a3));
				// c = s < 0 ? d : c;
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
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				// s = ~(a ^ b) & (c ^ a);
				xmm_s0 = _mm_andnot_si128(_mm_or_si128(xmm_a0, xmm_b), _mm_or_si128(xmm_c0, xmm_a0));
				// c = s < 0 ? d : c;
				xmm_s0 = _mm_srai_epi32(xmm_s0, 31);
				xmm_c0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_t), _mm_andnot_si128(xmm_s0, xmm_c0));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				r = a[i] + b;
				s = ~(a[i] ^ b) & (r ^ a[i]);
				c[i] = s >> 31 ? t : r;
			}
		}
	};

	template<>
	struct kernel_adds_value<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, const unsigned int *a, const unsigned int b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			unsigned int r;
			const __m128i xmm_max = _mm_set1_epi32(uint32_max);
			const __m128i xmm_b = _mm_set1_epi32(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi32(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi32(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi32(xmm_a3, xmm_b);
				// s = a > c;
				xmm_s0 = _mm_srai_epi32(_mm_xor_si128(xmm_a0, xmm_c0), 31);
				xmm_s1 = _mm_srai_epi32(_mm_xor_si128(xmm_a1, xmm_c1), 31);
				xmm_s2 = _mm_srai_epi32(_mm_xor_si128(xmm_a2, xmm_c2), 31);
				xmm_s3 = _mm_srai_epi32(_mm_xor_si128(xmm_a3, xmm_c3), 31);
				xmm_s0 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a0, xmm_c0), xmm_s0);
				xmm_s1 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a1, xmm_c1), xmm_s1);
				xmm_s2 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a2, xmm_c2), xmm_s2);
				xmm_s3 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a3, xmm_c3), xmm_s3);
				// c = s ? max : c;
				xmm_c0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_max), _mm_andnot_si128(xmm_s0, xmm_c0));
				xmm_c1 = _mm_or_si128(_mm_and_si128(xmm_s1, xmm_max), _mm_andnot_si128(xmm_s1, xmm_c1));
				xmm_c2 = _mm_or_si128(_mm_and_si128(xmm_s2, xmm_max), _mm_andnot_si128(xmm_s2, xmm_c2));
				xmm_c3 = _mm_or_si128(_mm_and_si128(xmm_s3, xmm_max), _mm_andnot_si128(xmm_s3, xmm_c3));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				// s = a > c;
				xmm_s0 = _mm_srai_epi32(_mm_xor_si128(xmm_a0, xmm_c0), 31);
				xmm_s0 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a0, xmm_c0), xmm_s0);
				// c = s ? max : c;
				xmm_c0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_max), _mm_andnot_si128(xmm_s0, xmm_c0));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				r = a[i] + b;
				c[i] = a[i] > r ? uint32_max : r;
			}
		}
	};

	template<>
	struct kernel_adds_value<signed char, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, const signed char b, signed char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_b = _mm256_set1_epi8(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi8(ymm_a0, ymm_b);
				ymm_c1 = _mm256_adds_epi8(ymm_a1, ymm_b);
				ymm_c2 = _mm256_adds_epi8(ymm_a2, ymm_b);
				ymm_c3 = _mm256_adds_epi8(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi8(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, const unsigned char *a, const unsigned char b, unsigned char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_b = _mm256_set1_epi8(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi8(ymm_a0, ymm_b);
				ymm_c1 = _mm256_adds_epi8(ymm_a1, ymm_b);
				ymm_c2 = _mm256_adds_epi8(ymm_a2, ymm_b);
				ymm_c3 = _mm256_adds_epi8(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi8(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed short *a, const signed short b, signed short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_b = _mm256_set1_epi16(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi16(ymm_a0, ymm_b);
				ymm_c1 = _mm256_adds_epi16(ymm_a1, ymm_b);
				ymm_c2 = _mm256_adds_epi16(ymm_a2, ymm_b);
				ymm_c3 = _mm256_adds_epi16(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi16(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const unsigned short *a, const unsigned short b, unsigned short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_b = _mm256_set1_epi16(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi16(ymm_a0, ymm_b);
				ymm_c1 = _mm256_adds_epi16(ymm_a1, ymm_b);
				ymm_c2 = _mm256_adds_epi16(ymm_a2, ymm_b);
				ymm_c3 = _mm256_adds_epi16(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = adds(a, b);
				ymm_c0 = _mm256_adds_epi16(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_adds_value<signed int, cpu_avx2>
	{
		void operator()(size_t n, const signed int *a, const signed int b, signed int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			signed int r, s;
			signed int t = b >> 31 ? int32_min : int32_max;
			const __m256i ymm_b = _mm256_set1_epi32(b);
			const __m256i ymm_t = _mm256_set1_epi32(t);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi32(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi32(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi32(ymm_a3, ymm_b);
				// s = ~(a ^ b) & (c ^ a);
				ymm_s0 = _mm256_andnot_si256(_mm256_or_si256(ymm_a0, ymm_b), _mm256_or_si256(ymm_c0, ymm_a0));
				ymm_s1 = _mm256_andnot_si256(_mm256_or_si256(ymm_a1, ymm_b), _mm256_or_si256(ymm_c1, ymm_a1));
				ymm_s2 = _mm256_andnot_si256(_mm256_or_si256(ymm_a2, ymm_b), _mm256_or_si256(ymm_c2, ymm_a2));
				ymm_s3 = _mm256_andnot_si256(_mm256_or_si256(ymm_a3, ymm_b), _mm256_or_si256(ymm_c3, ymm_a3));
				// c = s < 0 ? d : c;
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
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				// s = ~(a ^ b) & (c ^ a);
				ymm_s0 = _mm256_andnot_si256(_mm256_or_si256(ymm_a0, ymm_b), _mm256_or_si256(ymm_c0, ymm_a0));
				// c = s < 0 ? d : c;
				ymm_s0 = _mm256_srai_epi32(ymm_s0, 31);
				ymm_c0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_t), _mm256_andnot_si256(ymm_s0, ymm_c0));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				r = a[i] + b;
				s = ~(a[i] ^ b) & (r ^ a[i]);
				c[i] = s >> 31 ? t : r;
			}
		}
	};

	template<>
	struct kernel_adds_value<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, const unsigned int *a, const unsigned int b, unsigned int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			unsigned int r;
			const __m256i ymm_max = _mm256_set1_epi32(uint32_max);
			const __m256i ymm_b = _mm256_set1_epi32(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi32(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi32(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi32(ymm_a3, ymm_b);
				// s = a > c;
				ymm_s0 = _mm256_srai_epi32(_mm256_xor_si256(ymm_a0, ymm_c0), 31);
				ymm_s1 = _mm256_srai_epi32(_mm256_xor_si256(ymm_a1, ymm_c1), 31);
				ymm_s2 = _mm256_srai_epi32(_mm256_xor_si256(ymm_a2, ymm_c2), 31);
				ymm_s3 = _mm256_srai_epi32(_mm256_xor_si256(ymm_a3, ymm_c3), 31);
				ymm_s0 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a0, ymm_c0), ymm_s0);
				ymm_s1 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a1, ymm_c1), ymm_s1);
				ymm_s2 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a2, ymm_c2), ymm_s2);
				ymm_s3 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a3, ymm_c3), ymm_s3);
				// c = s ? max : c;
				ymm_c0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_max), _mm256_andnot_si256(ymm_s0, ymm_c0));
				ymm_c1 = _mm256_or_si256(_mm256_and_si256(ymm_s1, ymm_max), _mm256_andnot_si256(ymm_s1, ymm_c1));
				ymm_c2 = _mm256_or_si256(_mm256_and_si256(ymm_s2, ymm_max), _mm256_andnot_si256(ymm_s2, ymm_c2));
				ymm_c3 = _mm256_or_si256(_mm256_and_si256(ymm_s3, ymm_max), _mm256_andnot_si256(ymm_s3, ymm_c3));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				// s = a > c;
				ymm_s0 = _mm256_srai_epi32(_mm256_xor_si256(ymm_a0, ymm_c0), 31);
				ymm_s0 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a0, ymm_c0), ymm_s0);
				// c = s ? max : c;
				ymm_c0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_max), _mm256_andnot_si256(ymm_s0, ymm_c0));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				r = a[i] + b;
				c[i] = b > r ? uint32_max : r;
			}
		}
	};

} // namespace core

#endif
