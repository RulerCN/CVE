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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_ADDS_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_ADDS_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_adds

	template<class T, cpu_inst_type inst>
	struct kernel_adds
	{
		void operator()(size_t n, const T *a, T *b) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				b[0] = a[0] + b[0];
				b[1] = a[1] + b[1];
				b[2] = a[2] + b[2];
				b[3] = a[3] + b[3];
				b[4] = a[4] + b[4];
				b[5] = a[5] + b[5];
				b[6] = a[6] + b[6];
				b[7] = a[7] + b[7];
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a[i] + b[i];
		}
	};

	template<>
	struct kernel_adds<signed char, cpu_none>
	{
		void operator()(size_t n, const signed char *a, signed char *b) const
		{
			constexpr size_t block = 8;
			signed char c0, c1, c2, c3, c4, c5, c6, c7;
			signed char d0, d1, d2, d3, d4, d5, d6, d7;
			signed char s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a[0] + b[0];
				c1 = a[1] + b[1];
				c2 = a[2] + b[2];
				c3 = a[3] + b[3];
				c4 = a[4] + b[4];
				c5 = a[5] + b[5];
				c6 = a[6] + b[6];
				c7 = a[7] + b[7];
				d0 = a[0] >> 7 ? int8_min : int8_max;
				d1 = a[1] >> 7 ? int8_min : int8_max;
				d2 = a[2] >> 7 ? int8_min : int8_max;
				d3 = a[3] >> 7 ? int8_min : int8_max;
				d4 = a[4] >> 7 ? int8_min : int8_max;
				d5 = a[5] >> 7 ? int8_min : int8_max;
				d6 = a[6] >> 7 ? int8_min : int8_max;
				d7 = a[7] >> 7 ? int8_min : int8_max;
				s0 = ~(a[0] ^ b[0]) & (c0 ^ a[0]);
				s1 = ~(a[1] ^ b[1]) & (c1 ^ a[1]);
				s2 = ~(a[2] ^ b[2]) & (c2 ^ a[2]);
				s3 = ~(a[3] ^ b[3]) & (c3 ^ a[3]);
				s4 = ~(a[4] ^ b[4]) & (c4 ^ a[4]);
				s5 = ~(a[5] ^ b[5]) & (c5 ^ a[5]);
				s6 = ~(a[6] ^ b[6]) & (c6 ^ a[6]);
				s7 = ~(a[7] ^ b[7]) & (c7 ^ a[7]);
				b[0] = s0 >> 7 ? d0 : c0;
				b[1] = s1 >> 7 ? d1 : c1;
				b[2] = s2 >> 7 ? d2 : c2;
				b[3] = s3 >> 7 ? d3 : c3;
				b[4] = s4 >> 7 ? d4 : c4;
				b[5] = s5 >> 7 ? d5 : c5;
				b[6] = s6 >> 7 ? d6 : c6;
				b[7] = s7 >> 7 ? d7 : c7;
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b[i];
				d0 = a[i] >> 7 ? int8_min : int8_max;
				s0 = ~(a[i] ^ b[i]) & (c0 ^ a[i]);
				b[i] = s0 >> 7 ? d0 : c0;
			}
		}
	};

	template<>
	struct kernel_adds<unsigned char, cpu_none>
	{
		void operator()(size_t n, const unsigned char *a, unsigned char *b) const
		{
			constexpr size_t block = 8;
			unsigned char c0, c1, c2, c3, c4, c5, c6, c7;

			while (n > block)
			{
				c0 = a[0] + b[0];
				c1 = a[1] + b[1];
				c2 = a[2] + b[2];
				c3 = a[3] + b[3];
				c4 = a[4] + b[4];
				c5 = a[5] + b[5];
				c6 = a[6] + b[6];
				c7 = a[7] + b[7];
				b[0] = c0 < a[0] ? uint8_max : c0;
				b[1] = c1 < a[1] ? uint8_max : c1;
				b[2] = c2 < a[2] ? uint8_max : c2;
				b[3] = c3 < a[3] ? uint8_max : c3;
				b[4] = c4 < a[4] ? uint8_max : c4;
				b[5] = c5 < a[5] ? uint8_max : c5;
				b[6] = c6 < a[6] ? uint8_max : c6;
				b[7] = c7 < a[7] ? uint8_max : c7;
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b[i];
				b[i] = c0 < a[i] ? uint8_max : c0;
			}
		}
	};

	template<>
	struct kernel_adds<signed short, cpu_none>
	{
		void operator()(size_t n, const signed short *a, signed short *b) const
		{
			constexpr size_t block = 8;
			signed short c0, c1, c2, c3, c4, c5, c6, c7;
			signed short d0, d1, d2, d3, d4, d5, d6, d7;
			signed short s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a[0] + b[0];
				c1 = a[1] + b[1];
				c2 = a[2] + b[2];
				c3 = a[3] + b[3];
				c4 = a[4] + b[4];
				c5 = a[5] + b[5];
				c6 = a[6] + b[6];
				c7 = a[7] + b[7];
				d0 = a[0] >> 15 ? int16_min : int16_max;
				d1 = a[1] >> 15 ? int16_min : int16_max;
				d2 = a[2] >> 15 ? int16_min : int16_max;
				d3 = a[3] >> 15 ? int16_min : int16_max;
				d4 = a[4] >> 15 ? int16_min : int16_max;
				d5 = a[5] >> 15 ? int16_min : int16_max;
				d6 = a[6] >> 15 ? int16_min : int16_max;
				d7 = a[7] >> 15 ? int16_min : int16_max;
				s0 = ~(a[0] ^ b[0]) & (c0 ^ a[0]);
				s1 = ~(a[1] ^ b[1]) & (c1 ^ a[1]);
				s2 = ~(a[2] ^ b[2]) & (c2 ^ a[2]);
				s3 = ~(a[3] ^ b[3]) & (c3 ^ a[3]);
				s4 = ~(a[4] ^ b[4]) & (c4 ^ a[4]);
				s5 = ~(a[5] ^ b[5]) & (c5 ^ a[5]);
				s6 = ~(a[6] ^ b[6]) & (c6 ^ a[6]);
				s7 = ~(a[7] ^ b[7]) & (c7 ^ a[7]);
				b[0] = s0 >> 15 ? d0 : c0;
				b[1] = s1 >> 15 ? d1 : c1;
				b[2] = s2 >> 15 ? d2 : c2;
				b[3] = s3 >> 15 ? d3 : c3;
				b[4] = s4 >> 15 ? d4 : c4;
				b[5] = s5 >> 15 ? d5 : c5;
				b[6] = s6 >> 15 ? d6 : c6;
				b[7] = s7 >> 15 ? d7 : c7;
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b[i];
				d0 = a[i] >> 15 ? int16_min : int16_max;
				s0 = ~(a[i] ^ b[i]) & (c0 ^ a[i]);
				b[i] = s0 >> 15 ? d0 : c0;
			}
		}
	};

	template<>
	struct kernel_adds<unsigned short, cpu_none>
	{
		void operator()(size_t n, const unsigned short *a, unsigned short *b) const
		{
			constexpr size_t block = 8;
			unsigned short c0, c1, c2, c3, c4, c5, c6, c7;

			while (n > block)
			{
				c0 = a[0] + b[0];
				c1 = a[1] + b[1];
				c2 = a[2] + b[2];
				c3 = a[3] + b[3];
				c4 = a[4] + b[4];
				c5 = a[5] + b[5];
				c6 = a[6] + b[6];
				c7 = a[7] + b[7];
				b[0] = c0 < a[0] ? uint16_max : c0;
				b[1] = c1 < a[1] ? uint16_max : c1;
				b[2] = c2 < a[2] ? uint16_max : c2;
				b[3] = c3 < a[3] ? uint16_max : c3;
				b[4] = c4 < a[4] ? uint16_max : c4;
				b[5] = c5 < a[5] ? uint16_max : c5;
				b[6] = c6 < a[6] ? uint16_max : c6;
				b[7] = c7 < a[7] ? uint16_max : c7;
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b[i];
				b[i] = c0 < a[i] ? uint16_max : c0;
			}
		}
	};

	template<>
	struct kernel_adds<signed int, cpu_none>
	{
		void operator()(size_t n, const signed int *a, signed int *b) const
		{
			constexpr size_t block = 8;
			signed int c0, c1, c2, c3, c4, c5, c6, c7;
			signed int d0, d1, d2, d3, d4, d5, d6, d7;
			signed int s0, s1, s2, s3, s4, s5, s6, s7;

			while (n > block)
			{
				c0 = a[0] + b[0];
				c1 = a[1] + b[1];
				c2 = a[2] + b[2];
				c3 = a[3] + b[3];
				c4 = a[4] + b[4];
				c5 = a[5] + b[5];
				c6 = a[6] + b[6];
				c7 = a[7] + b[7];
				d0 = a[0] >> 31 ? int32_min : int32_max;
				d1 = a[1] >> 31 ? int32_min : int32_max;
				d2 = a[2] >> 31 ? int32_min : int32_max;
				d3 = a[3] >> 31 ? int32_min : int32_max;
				d4 = a[4] >> 31 ? int32_min : int32_max;
				d5 = a[5] >> 31 ? int32_min : int32_max;
				d6 = a[6] >> 31 ? int32_min : int32_max;
				d7 = a[7] >> 31 ? int32_min : int32_max;
				s0 = ~(a[0] ^ b[0]) & (c0 ^ a[0]);
				s1 = ~(a[1] ^ b[1]) & (c1 ^ a[1]);
				s2 = ~(a[2] ^ b[2]) & (c2 ^ a[2]);
				s3 = ~(a[3] ^ b[3]) & (c3 ^ a[3]);
				s4 = ~(a[4] ^ b[4]) & (c4 ^ a[4]);
				s5 = ~(a[5] ^ b[5]) & (c5 ^ a[5]);
				s6 = ~(a[6] ^ b[6]) & (c6 ^ a[6]);
				s7 = ~(a[7] ^ b[7]) & (c7 ^ a[7]);
				b[0] = s0 >> 31 ? d0 : c0;
				b[1] = s1 >> 31 ? d1 : c1;
				b[2] = s2 >> 31 ? d2 : c2;
				b[3] = s3 >> 31 ? d3 : c3;
				b[4] = s4 >> 31 ? d4 : c4;
				b[5] = s5 >> 31 ? d5 : c5;
				b[6] = s6 >> 31 ? d6 : c6;
				b[7] = s7 >> 31 ? d7 : c7;
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b[i];
				d0 = a[i] >> 31 ? int32_min : int32_max;
				s0 = ~(a[i] ^ b[i]) & (c0 ^ a[i]);
				b[i] = s0 >> 31 ? d0 : c0;
			}
		}
	};

	template<>
	struct kernel_adds<unsigned int, cpu_none>
	{
		void operator()(size_t n, const unsigned int *a, unsigned int *b) const
		{
			constexpr size_t block = 8;
			unsigned int c0, c1, c2, c3, c4, c5, c6, c7;

			while (n > block)
			{
				c0 = a[0] + b[0];
				c1 = a[1] + b[1];
				c2 = a[2] + b[2];
				c3 = a[3] + b[3];
				c4 = a[4] + b[4];
				c5 = a[5] + b[5];
				c6 = a[6] + b[6];
				c7 = a[7] + b[7];
				b[0] = c0 < a[0] ? uint32_max : c0;
				b[1] = c1 < a[1] ? uint32_max : c1;
				b[2] = c2 < a[2] ? uint32_max : c2;
				b[3] = c3 < a[3] ? uint32_max : c3;
				b[4] = c4 < a[4] ? uint32_max : c4;
				b[5] = c5 < a[5] ? uint32_max : c5;
				b[6] = c6 < a[6] ? uint32_max : c6;
				b[7] = c7 < a[7] ? uint32_max : c7;
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c0 = a[i] + b[i];
				b[i] = c0 < a[i] ? uint32_max : c0;
			}
		}
	};

	template<>
	struct kernel_adds<signed char, cpu_sse2>
	{
		void operator()(size_t n, const signed char *a, signed char *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epi8(xmm_a0, xmm_b0);
				xmm_b1 = _mm_adds_epi8(xmm_a1, xmm_b1);
				xmm_b2 = _mm_adds_epi8(xmm_a2, xmm_b2);
				xmm_b3 = _mm_adds_epi8(xmm_a3, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epi8(xmm_a0, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_a0 = _mm_set1_epi8(a[i]);
				xmm_b0 = _mm_set1_epi8(b[i]);
				xmm_b0 = _mm_adds_epi8(xmm_a0, xmm_b0);
				b[i] = reinterpret_cast<signed char*>(&xmm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<unsigned char, cpu_sse2>
	{
		void operator()(size_t n, const unsigned char *a, unsigned char *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epu8(xmm_a0, xmm_b0);
				xmm_b1 = _mm_adds_epu8(xmm_a1, xmm_b1);
				xmm_b2 = _mm_adds_epu8(xmm_a2, xmm_b2);
				xmm_b3 = _mm_adds_epu8(xmm_a3, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epu8(xmm_a0, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_a0 = _mm_set1_epi8(a[i]);
				xmm_b0 = _mm_set1_epi8(b[i]);
				xmm_b0 = _mm_adds_epu8(xmm_a0, xmm_b0);
				b[i] = reinterpret_cast<unsigned char*>(&xmm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<signed short, cpu_sse2>
	{
		void operator()(size_t n, const signed short *a, signed short *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epi16(xmm_a0, xmm_b0);
				xmm_b1 = _mm_adds_epi16(xmm_a1, xmm_b1);
				xmm_b2 = _mm_adds_epi16(xmm_a2, xmm_b2);
				xmm_b3 = _mm_adds_epi16(xmm_a3, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epi16(xmm_a0, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_a0 = _mm_set1_epi16(a[i]);
				xmm_b0 = _mm_set1_epi16(b[i]);
				xmm_b0 = _mm_adds_epi16(xmm_a0, xmm_b0);
				b[i] = reinterpret_cast<signed short*>(&xmm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<unsigned short, cpu_sse2>
	{
		void operator()(size_t n, const unsigned short *a, unsigned short *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epu16(xmm_a0, xmm_b0);
				xmm_b1 = _mm_adds_epu16(xmm_a1, xmm_b1);
				xmm_b2 = _mm_adds_epu16(xmm_a2, xmm_b2);
				xmm_b3 = _mm_adds_epu16(xmm_a3, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = adds(a, b);
				xmm_b0 = _mm_adds_epu16(xmm_a0, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				xmm_a0 = _mm_set1_epi16(a[i]);
				xmm_b0 = _mm_set1_epi16(b[i]);
				xmm_b0 = _mm_adds_epu16(xmm_a0, xmm_b0);
				b[i] = reinterpret_cast<unsigned short*>(&xmm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<signed int, cpu_sse2>
	{
		void operator()(size_t n, const signed int *a, signed int *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_min = _mm_set1_epi32(int32_min);
			const __m128i xmm_max = _mm_set1_epi32(int32_max);
			const __m128i xmm_sign = _mm_set1_epi32(int32_sign);
			signed int c, d, s;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;
			__m128i xmm_d0, xmm_d1, xmm_d2, xmm_d3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b0);
				xmm_c1 = _mm_add_epi32(xmm_a1, xmm_b1);
				xmm_c2 = _mm_add_epi32(xmm_a2, xmm_b2);
				xmm_c3 = _mm_add_epi32(xmm_a3, xmm_b3);
				// d = a < 0 ? min : max;
				xmm_a0 = _mm_srai_epi32(xmm_a0, 31);
				xmm_a1 = _mm_srai_epi32(xmm_a1, 31);
				xmm_a2 = _mm_srai_epi32(xmm_a2, 31);
				xmm_a3 = _mm_srai_epi32(xmm_a3, 31);
				xmm_d0 = _mm_or_si128(_mm_and_si128(xmm_a0, xmm_min), _mm_andnot_si128(xmm_a0, xmm_max));
				xmm_d1 = _mm_or_si128(_mm_and_si128(xmm_a1, xmm_min), _mm_andnot_si128(xmm_a1, xmm_max));
				xmm_d2 = _mm_or_si128(_mm_and_si128(xmm_a2, xmm_min), _mm_andnot_si128(xmm_a2, xmm_max));
				xmm_d3 = _mm_or_si128(_mm_and_si128(xmm_a3, xmm_min), _mm_andnot_si128(xmm_a3, xmm_max));
				// s = ~(a ^ b) & (c ^ a);
				xmm_s0 = _mm_andnot_si128(_mm_or_si128(xmm_a0, xmm_b0), _mm_or_si128(xmm_c0, xmm_a0));
				xmm_s1 = _mm_andnot_si128(_mm_or_si128(xmm_a1, xmm_b1), _mm_or_si128(xmm_c1, xmm_a1));
				xmm_s2 = _mm_andnot_si128(_mm_or_si128(xmm_a2, xmm_b2), _mm_or_si128(xmm_c2, xmm_a2));
				xmm_s3 = _mm_andnot_si128(_mm_or_si128(xmm_a3, xmm_b3), _mm_or_si128(xmm_c3, xmm_a3));
				// b = s < 0 ? d : c;
				xmm_s0 = _mm_srai_epi32(xmm_s0, 31);
				xmm_s1 = _mm_srai_epi32(xmm_s1, 31);
				xmm_s2 = _mm_srai_epi32(xmm_s2, 31);
				xmm_s3 = _mm_srai_epi32(xmm_s3, 31);
				xmm_b0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_d0), _mm_andnot_si128(xmm_s0, xmm_c0));
				xmm_b1 = _mm_or_si128(_mm_and_si128(xmm_s1, xmm_d1), _mm_andnot_si128(xmm_s1, xmm_c1));
				xmm_b2 = _mm_or_si128(_mm_and_si128(xmm_s2, xmm_d2), _mm_andnot_si128(xmm_s2, xmm_c2));
				xmm_b3 = _mm_or_si128(_mm_and_si128(xmm_s3, xmm_d3), _mm_andnot_si128(xmm_s3, xmm_c3));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b0);
				// d = a < 0 ? min : max;
				xmm_a0 = _mm_srai_epi32(xmm_a0, 31);
				xmm_d0 = _mm_or_si128(_mm_and_si128(xmm_a0, xmm_min), _mm_andnot_si128(xmm_a0, xmm_max));
				// s = ~(a ^ b) & (c ^ a);
				xmm_s0 = _mm_andnot_si128(_mm_or_si128(xmm_a0, xmm_b0), _mm_or_si128(xmm_c0, xmm_a0));
				// b = s < 0 ? d : c;
				xmm_s0 = _mm_srai_epi32(xmm_s0, 31);
				xmm_b0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_d0), _mm_andnot_si128(xmm_s0, xmm_c0));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c = a[i] + b[i];
				d = a[i] >> 31 ? int32_min : int32_max;
				s = ~(a[i] ^ b[i]) & (c ^ a[i]);
				b[i] = s >> 31 ? d : c;
			}
		}
	};

	template<>
	struct kernel_adds<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, const unsigned int *a, unsigned int *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_max = _mm_set1_epi32(uint32_max);
			unsigned int c;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b0);
				xmm_c1 = _mm_add_epi32(xmm_a1, xmm_b1);
				xmm_c2 = _mm_add_epi32(xmm_a2, xmm_b2);
				xmm_c3 = _mm_add_epi32(xmm_a3, xmm_b3);
				// b = c < a ? max : c;
				xmm_s0 = _mm_cmplt_epi32(xmm_c0, xmm_a0);
				xmm_s1 = _mm_cmplt_epi32(xmm_c1, xmm_a1);
				xmm_s2 = _mm_cmplt_epi32(xmm_c2, xmm_a2);
				xmm_s3 = _mm_cmplt_epi32(xmm_c3, xmm_a3);
				xmm_b0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_max), _mm_andnot_si128(xmm_s0, xmm_c0));
				xmm_b1 = _mm_or_si128(_mm_and_si128(xmm_s1, xmm_max), _mm_andnot_si128(xmm_s1, xmm_c1));
				xmm_b2 = _mm_or_si128(_mm_and_si128(xmm_s2, xmm_max), _mm_andnot_si128(xmm_s2, xmm_c2));
				xmm_b3 = _mm_or_si128(_mm_and_si128(xmm_s3, xmm_max), _mm_andnot_si128(xmm_s3, xmm_c3));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b0);
				// b = c < a ? max : c;
				xmm_s0 = _mm_cmplt_epi32(xmm_c0, xmm_a0);
				xmm_b0 = _mm_or_si128(_mm_and_si128(xmm_s0, xmm_max), _mm_andnot_si128(xmm_s0, xmm_c0));
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c = a[i] + b[i];
				b[i] = c < a[i] ? uint32_max : c;
			}
		}
	};

	template<>
	struct kernel_adds<signed char, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, signed char *b) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epi8(ymm_a0, ymm_b0);
				ymm_b1 = _mm256_adds_epi8(ymm_a1, ymm_b1);
				ymm_b2 = _mm256_adds_epi8(ymm_a2, ymm_b2);
				ymm_b3 = _mm256_adds_epi8(ymm_a3, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epi8(ymm_a0, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_a0 = _mm256_set1_epi8(a[i]);
				ymm_b0 = _mm256_set1_epi8(b[i]);
				ymm_b0 = _mm256_adds_epi8(ymm_a0, ymm_b0);
				b[i] = reinterpret_cast<signed char*>(&ymm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, const unsigned char *a, unsigned char *b) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epu8(ymm_a0, ymm_b0);
				ymm_b1 = _mm256_adds_epu8(ymm_a1, ymm_b1);
				ymm_b2 = _mm256_adds_epu8(ymm_a2, ymm_b2);
				ymm_b3 = _mm256_adds_epu8(ymm_a3, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epu8(ymm_a0, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_a0 = _mm256_set1_epi8(a[i]);
				ymm_b0 = _mm256_set1_epi8(b[i]);
				ymm_b0 = _mm256_adds_epu8(ymm_a0, ymm_b0);
				b[i] = reinterpret_cast<unsigned char*>(&ymm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed short *a, signed short *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epi16(ymm_a0, ymm_b0);
				ymm_b1 = _mm256_adds_epi16(ymm_a1, ymm_b1);
				ymm_b2 = _mm256_adds_epi16(ymm_a2, ymm_b2);
				ymm_b3 = _mm256_adds_epi16(ymm_a3, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epi16(ymm_a0, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_a0 = _mm256_set1_epi16(a[i]);
				ymm_b0 = _mm256_set1_epi16(b[i]);
				ymm_b0 = _mm256_adds_epi16(ymm_a0, ymm_b0);
				b[i] = reinterpret_cast<signed short*>(&ymm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const unsigned short *a, unsigned short *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epu16(ymm_a0, ymm_b0);
				ymm_b1 = _mm256_adds_epu16(ymm_a1, ymm_b1);
				ymm_b2 = _mm256_adds_epu16(ymm_a2, ymm_b2);
				ymm_b3 = _mm256_adds_epu16(ymm_a3, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = adds(a, b);
				ymm_b0 = _mm256_adds_epu16(ymm_a0, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				ymm_a0 = _mm256_set1_epi16(a[i]);
				ymm_b0 = _mm256_set1_epi16(b[i]);
				ymm_b0 = _mm256_adds_epu16(ymm_a0, ymm_b0);
				b[i] = reinterpret_cast<unsigned short*>(&ymm_b0)[0];
			}
		}
	};

	template<>
	struct kernel_adds<signed int, cpu_avx2>
	{
		void operator()(size_t n, const signed int *a, signed int *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_min = _mm256_set1_epi32(int32_min);
			const __m256i ymm_max = _mm256_set1_epi32(int32_max);
			const __m256i ymm_sign = _mm256_set1_epi32(int32_sign);
			signed int c, d, s;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;
			__m256i ymm_d0, ymm_d1, ymm_d2, ymm_d3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_add_epi32(ymm_a1, ymm_b1);
				ymm_c2 = _mm256_add_epi32(ymm_a2, ymm_b2);
				ymm_c3 = _mm256_add_epi32(ymm_a3, ymm_b3);
				// d = a < 0 ? min : max;
				ymm_a0 = _mm256_srai_epi32(ymm_a0, 31);
				ymm_a1 = _mm256_srai_epi32(ymm_a1, 31);
				ymm_a2 = _mm256_srai_epi32(ymm_a2, 31);
				ymm_a3 = _mm256_srai_epi32(ymm_a3, 31);
				ymm_d0 = _mm256_or_si256(_mm256_and_si256(ymm_a0, ymm_min), _mm256_andnot_si256(ymm_a0, ymm_max));
				ymm_d1 = _mm256_or_si256(_mm256_and_si256(ymm_a1, ymm_min), _mm256_andnot_si256(ymm_a1, ymm_max));
				ymm_d2 = _mm256_or_si256(_mm256_and_si256(ymm_a2, ymm_min), _mm256_andnot_si256(ymm_a2, ymm_max));
				ymm_d3 = _mm256_or_si256(_mm256_and_si256(ymm_a3, ymm_min), _mm256_andnot_si256(ymm_a3, ymm_max));
				// s = ~(a ^ b) & (c ^ a);
				ymm_s0 = _mm256_andnot_si256(_mm256_or_si256(ymm_a0, ymm_b0), _mm256_or_si256(ymm_c0, ymm_a0));
				ymm_s1 = _mm256_andnot_si256(_mm256_or_si256(ymm_a1, ymm_b1), _mm256_or_si256(ymm_c1, ymm_a1));
				ymm_s2 = _mm256_andnot_si256(_mm256_or_si256(ymm_a2, ymm_b2), _mm256_or_si256(ymm_c2, ymm_a2));
				ymm_s3 = _mm256_andnot_si256(_mm256_or_si256(ymm_a3, ymm_b3), _mm256_or_si256(ymm_c3, ymm_a3));
				// b = s < 0 ? d : c;
				ymm_s0 = _mm256_srai_epi32(ymm_s0, 31);
				ymm_s1 = _mm256_srai_epi32(ymm_s1, 31);
				ymm_s2 = _mm256_srai_epi32(ymm_s2, 31);
				ymm_s3 = _mm256_srai_epi32(ymm_s3, 31);
				ymm_b0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_d0), _mm256_andnot_si256(ymm_s0, ymm_c0));
				ymm_b1 = _mm256_or_si256(_mm256_and_si256(ymm_s1, ymm_d1), _mm256_andnot_si256(ymm_s1, ymm_c1));
				ymm_b2 = _mm256_or_si256(_mm256_and_si256(ymm_s2, ymm_d2), _mm256_andnot_si256(ymm_s2, ymm_c2));
				ymm_b3 = _mm256_or_si256(_mm256_and_si256(ymm_s3, ymm_d3), _mm256_andnot_si256(ymm_s3, ymm_c3));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b0);
				// d = a < 0 ? min : max;
				ymm_a0 = _mm256_srai_epi32(ymm_a0, 31);
				ymm_d0 = _mm256_or_si256(_mm256_and_si256(ymm_a0, ymm_min), _mm256_andnot_si256(ymm_a0, ymm_max));
				// s = ~(a ^ b) & (c ^ a);
				ymm_s0 = _mm256_andnot_si256(_mm256_or_si256(ymm_a0, ymm_b0), _mm256_or_si256(ymm_c0, ymm_a0));
				// b = s < 0 ? d : c;
				ymm_s0 = _mm256_srai_epi32(ymm_s0, 31);
				ymm_b0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_d0), _mm256_andnot_si256(ymm_s0, ymm_c0));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c = a[i] + b[i];
				d = a[i] >> 31 ? int32_min : int32_max;
				s = ~(a[i] ^ b[i]) & (c ^ a[i]);
				b[i] = s >> 31 ? d : c;
			}
		}
	};

	template<>
	struct kernel_adds<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, const unsigned int *a, unsigned int *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_max = _mm256_set1_epi32(uint32_max);
			unsigned int c;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_add_epi32(ymm_a1, ymm_b1);
				ymm_c2 = _mm256_add_epi32(ymm_a2, ymm_b2);
				ymm_c3 = _mm256_add_epi32(ymm_a3, ymm_b3);
				// b = c < a ? max : c;
				ymm_s0 = _mm256_cmpgt_epi32(ymm_a0, ymm_c0);
				ymm_s1 = _mm256_cmpgt_epi32(ymm_a1, ymm_c1);
				ymm_s2 = _mm256_cmpgt_epi32(ymm_a2, ymm_c2);
				ymm_s3 = _mm256_cmpgt_epi32(ymm_a3, ymm_c3);
				ymm_b0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_max), _mm256_andnot_si256(ymm_s0, ymm_c0));
				ymm_b1 = _mm256_or_si256(_mm256_and_si256(ymm_s1, ymm_max), _mm256_andnot_si256(ymm_s1, ymm_c1));
				ymm_b2 = _mm256_or_si256(_mm256_and_si256(ymm_s2, ymm_max), _mm256_andnot_si256(ymm_s2, ymm_c2));
				ymm_b3 = _mm256_or_si256(_mm256_and_si256(ymm_s3, ymm_max), _mm256_andnot_si256(ymm_s3, ymm_c3));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b0);
				// b = c < a ? max : c;
				ymm_s0 = _mm256_cmpgt_epi32(ymm_a0, ymm_c0);
				ymm_b0 = _mm256_or_si256(_mm256_and_si256(ymm_s0, ymm_max), _mm256_andnot_si256(ymm_s0, ymm_c0));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
			{
				c = a[i] + b[i];
				b[i] = c < a[i] ? uint32_max : c;
			}
		}
	};

} // namespace core

#endif
