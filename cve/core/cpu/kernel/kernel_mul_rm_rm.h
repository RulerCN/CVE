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

#ifndef __CORE_CPU_KERNEL_MUL_RM_RM_H__
#define __CORE_CPU_KERNEL_MUL_RM_RM_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template common_mul_rm_rm
	template<class T>
	struct common_mul_rm_rm
	{
		// C(mxn) += A(mxp) * B(pxn)
		void operator()(size_t m, size_t p, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b;
			T val_a;

			for (int i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (int k = 0; k < p; ++k)
				{
					val_a = a[k];
					for (int j = 0; j < n; ++j)
						c[j] += val_a * ptr_b[j];
					ptr_b += rsb;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	// Class template block_mul_rv_rm
	template<class T, cpu_inst_type inst>
	struct block_mul_rm_rm
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b0 = b;
			const T *ptr_b1 = ptr_b0 + rsb;
			const T *ptr_b2 = ptr_b1 + rsb;
			const T *ptr_b3 = ptr_b2 + rsb;
			T val_a0, val_a1, val_a2, val_a3;
			T val_c0, val_c1, val_c2, val_c3;

			for (size_t i = 0; i < 4; ++i)
			{
				val_a0 = a[0];
				val_a1 = a[1];
				val_a2 = a[2];
				val_a3 = a[3];
				for (size_t j = 0; j < n; ++j)
				{
					val_c0 = val_a0 * ptr_b0[j];
					val_c1 = val_a1 * ptr_b1[j];
					val_c2 = val_a2 * ptr_b2[j];
					val_c3 = val_a3 * ptr_b3[j];
					val_c0 = (val_c0 + val_c1) + (val_c2 + val_c3);
					c[j] += val_c0;
				}
				a += rsa;
				c += rsc;
			}
		}
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_a0 = a;
			const T *ptr_a1 = ptr_a0 + rsa;
			const T *ptr_a2 = ptr_a1 + rsa;
			const T *ptr_a3 = ptr_a2 + rsa;
			T *ptr_c0 = c;
			T *ptr_c1 = ptr_c0 + rsc;
			T *ptr_c2 = ptr_c1 + rsc;
			T *ptr_c3 = ptr_c2 + rsc;
			T val_a0, val_a1, val_a2, val_a3;
			T val_c0, val_c1, val_c2, val_c3;

			for (size_t k = aligned_p; k < p; ++k)
			{
				val_a0 = ptr_a0[k];
				val_a1 = ptr_a1[k];
				val_a2 = ptr_a2[k];
				val_a3 = ptr_a3[k];
				for (size_t j = 0; j < n; ++j)
				{
					ptr_c0[j] += val_a0 * b[j];
					ptr_c1[j] += val_a1 * b[j];
					ptr_c2[j] += val_a2 * b[j];
					ptr_c3[j] += val_a3 * b[j];
				}
				b += rsb;
			}
		}
		// C(mxn) += A(mx4) * B(4xn)
		void operator()(size_t m, size_t, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b0 = b;
			const T *ptr_b1 = ptr_b0 + rsb;
			const T *ptr_b2 = ptr_b1 + rsb;
			const T *ptr_b3 = ptr_b2 + rsb;
			T val_a0, val_a1, val_a2, val_a3;
			T val_c0, val_c1, val_c2, val_c3;

			for (size_t i = 0; i < m; ++i)
			{
				val_a0 = a[0];
				val_a1 = a[1];
				val_a2 = a[2];
				val_a3 = a[3];
				for (size_t j = 0; j < n; ++j)
				{
					val_c0 = val_a0 * ptr_b0[j];
					val_c1 = val_a1 * ptr_b1[j];
					val_c2 = val_a2 * ptr_b2[j];
					val_c3 = val_a3 * ptr_b3[j];
					val_c0 = (val_c0 + val_c1) + (val_c2 + val_c3);
					c[j] += val_c0;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rm_rm<float, cpu_sse3>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[4] = { a };
			const float *ptr_b[4] = { b };
			float *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					xmm_a0 = _mm_set1_ps(a[0]);
					xmm_a1 = _mm_set1_ps(a[1]);
					xmm_a2 = _mm_set1_ps(a[2]);
					xmm_a3 = _mm_set1_ps(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_ps(ptr_b[1] + j);
						xmm_b2 = _mm_loadu_ps(ptr_b[2] + j);
						xmm_b3 = _mm_loadu_ps(ptr_b[3] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
						xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b2);
						xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b3);
						xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
						xmm_c2 = _mm_add_ps(xmm_c2, xmm_c3);
						xmm_c0 = _mm_add_ps(xmm_c0, xmm_c2);
						// store data into memory
						_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_ps(ptr_a[0]);
				xmm_a1 = _mm_loadu_ps(ptr_a[1]);
				xmm_a2 = _mm_loadu_ps(ptr_a[2]);
				xmm_a3 = _mm_loadu_ps(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_ps(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b0);
					xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b0);
					xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b0);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c1);
					xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c3);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c2);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
				}
			}
		}
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[4] = { a };
			const float *ptr_b[4] = { b };
			float *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				//for (size_t k = aligned_p; k < p; ++k)
				for (size_t k = 0; k < p - aligned_p; ++k)
				{
					//xmm_a0 = _mm_set1_ps(ptr_a[0][k]);
					//xmm_a1 = _mm_set1_ps(ptr_a[1][k]);
					//xmm_a2 = _mm_set1_ps(ptr_a[2][k]);
					//xmm_a3 = _mm_set1_ps(ptr_a[3][k]);
					xmm_a0 = _mm_set_ps(ptr_a[3][k], ptr_a[2][k], ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b[k] + j);
						//xmm_b1 = _mm_loadu_ps(ptr_b[1] + j);
						//xmm_b2 = _mm_loadu_ps(ptr_b[2] + j);
						//xmm_b3 = _mm_loadu_ps(ptr_b[3] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						//xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b0);
						//xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b0);
						//xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b0);
						//xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
						//xmm_c2 = _mm_add_ps(xmm_c2, xmm_c3);
						//xmm_c0 = _mm_add_ps(xmm_c0, xmm_c2);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
					}
				}
			}
			if (aligned_n < n)
			{
				//for (size_t k = aligned_p; k < p; ++k)
				for (size_t k = 0; k < p - aligned_p; ++k)
				{
					xmm_a0 = _mm_set_ps(ptr_a[3][k], ptr_a[2][k], ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set1_ps(ptr_b[k][j]);
						// return the product
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
					}
				}
			}
		}
		// C(mxn) += A(mx4) * B(4xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[4] = { a };
			const float *ptr_b[4] = { b };
			float *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < m; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					xmm_a0 = _mm_set1_ps(a[0]);
					xmm_a1 = _mm_set1_ps(a[1]);
					xmm_a2 = _mm_set1_ps(a[2]);
					xmm_a3 = _mm_set1_ps(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_ps(ptr_b[1] + j);
						xmm_b2 = _mm_loadu_ps(ptr_b[2] + j);
						xmm_b3 = _mm_loadu_ps(ptr_b[3] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
						xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b2);
						xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b3);
						xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
						xmm_c2 = _mm_add_ps(xmm_c2, xmm_c3);
						xmm_c0 = _mm_add_ps(xmm_c0, xmm_c2);
						// store data into memory
						_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t i = 0; i < m; ++i)
				{
					xmm_a0 = _mm_loadu_ps(ptr_a[i]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set_ps(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
						// return the product
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
						xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
						// store data into memory
						ptr_c[i][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					}
				}
			}
		}
	};

	template<>
	struct block_mul_rm_rm<float, cpu_sse3 | cpu_fma>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[4] = { a };
			const float *ptr_b[4] = { b };
			float *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					xmm_a0 = _mm_set1_ps(a[0]);
					xmm_a1 = _mm_set1_ps(a[1]);
					xmm_a2 = _mm_set1_ps(a[2]);
					xmm_a3 = _mm_set1_ps(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_ps(ptr_b[1] + j);
						xmm_b2 = _mm_loadu_ps(ptr_b[2] + j);
						xmm_b3 = _mm_loadu_ps(ptr_b[3] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
						xmm_c0 = _mm_fmadd_ps(xmm_a2, xmm_b2, xmm_c0);
						xmm_c1 = _mm_fmadd_ps(xmm_a3, xmm_b3, xmm_c1);
						xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
						// store data into memory
						_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_ps(ptr_a[0]);
				xmm_a1 = _mm_loadu_ps(ptr_a[1]);
				xmm_a2 = _mm_loadu_ps(ptr_a[2]);
				xmm_a3 = _mm_loadu_ps(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_ps(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b0);
					xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b0);
					xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b0);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c1);
					xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c3);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c2);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
				}
			}
		}
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[4] = { a };
			const float *ptr_b[4] = { b };
			float *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					xmm_a0 = _mm_set1_ps(ptr_a[0][k]);
					xmm_a1 = _mm_set1_ps(ptr_a[1][k]);
					xmm_a2 = _mm_set1_ps(ptr_a[2][k]);
					xmm_a3 = _mm_set1_ps(ptr_a[3][k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_ps(ptr_b[1] + j);
						xmm_b2 = _mm_loadu_ps(ptr_b[2] + j);
						xmm_b3 = _mm_loadu_ps(ptr_b[3] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
						xmm_c0 = _mm_fmadd_ps(xmm_a2, xmm_b2, xmm_c0);
						xmm_c1 = _mm_fmadd_ps(xmm_a3, xmm_b3, xmm_c1);
						xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					xmm_a0 = _mm_set_ps(ptr_a[3][k], ptr_a[2][k], ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set1_ps(ptr_b[k][j]);
						// return the product
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
					}
				}
			}
		}
		// C(mxn) += A(mx4) * B(4xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[4] = { a };
			const float *ptr_b[4] = { b };
			float *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < m; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					xmm_a0 = _mm_set1_ps(a[0]);
					xmm_a1 = _mm_set1_ps(a[1]);
					xmm_a2 = _mm_set1_ps(a[2]);
					xmm_a3 = _mm_set1_ps(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_ps(ptr_b[1] + j);
						xmm_b2 = _mm_loadu_ps(ptr_b[2] + j);
						xmm_b3 = _mm_loadu_ps(ptr_b[3] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
						xmm_c0 = _mm_fmadd_ps(xmm_a2, xmm_b2, xmm_c0);
						xmm_c1 = _mm_fmadd_ps(xmm_a3, xmm_b3, xmm_c1);
						xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
						// store data into memory
						_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t i = 0; i < m; ++i)
				{
					xmm_a0 = _mm_loadu_ps(ptr_a[i]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set_ps(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
						// return the product
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
						xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
						// store data into memory
						ptr_c[i][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					}
				}
			}
		}
	};

	template<>
	struct block_mul_rm_rm<double, cpu_sse3>
	{
		// C(2xn) += A(2x2) * B(2xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[2] = { a, a + rsa };
			const double *ptr_b[2] = { b, b + rsb };
			double *ptr_c[2] = { c, c + rsc };
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 2; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					xmm_a0 = _mm_set1_pd(a[0]);
					xmm_a1 = _mm_set1_pd(a[1]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_pd(ptr_b[1] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b1);
						xmm_c0 = _mm_add_pd(xmm_c0, xmm_c1);
						// store data into memory
						_mm_storeu_pd(c, _mm_add_pd(_mm_loadu_pd(c), xmm_c0));
						c += 2;
					}
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_pd(ptr_a[0]);
				xmm_a1 = _mm_loadu_pd(ptr_a[1]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_pd(ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b0);
					xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<double*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<double*>(&xmm_c0)[1];
				}
			}
		}
		// C(2xn) += A(2xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[2] = { a, a + rsa };
			const double *ptr_b[2] = { b, b + rsb };
			double *ptr_c[2] = { c, c + rsc };
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					xmm_a0 = _mm_set1_pd(ptr_a[0][k]);
					xmm_a1 = _mm_set1_pd(ptr_a[1][k]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_pd(ptr_b[1] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b1);
						xmm_c0 = _mm_add_pd(xmm_c0, xmm_c1);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<double*>(&xmm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<double*>(&xmm_c0)[1];
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					xmm_a0 = _mm_set_pd(ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set1_pd(ptr_b[k][j]);
						// return the product
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<double*>(&xmm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<double*>(&xmm_c0)[1];
					}
				}
			}
		}
		// C(mxn) += A(mx2) * B(2xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[2] = { a, a + rsa };
			const double *ptr_b[2] = { b, b + rsb };
			double *ptr_c[2] = { c, c + rsc };
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < m; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					xmm_a0 = _mm_set1_pd(a[0]);
					xmm_a1 = _mm_set1_pd(a[1]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_pd(ptr_b[1] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b1);
						xmm_c0 = _mm_add_pd(xmm_c0, xmm_c1);
						// store data into memory
						_mm_storeu_pd(c, _mm_add_pd(_mm_loadu_pd(c), xmm_c0));
						c += 2;
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t i = 0; i < m; ++i)
				{
					xmm_a0 = _mm_loadu_pd(ptr_a[i]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set_pd(ptr_b[1][j], ptr_b[0][j]);
						// return the product
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c0);
						// store data into memory
						ptr_c[i][j] += reinterpret_cast<double*>(&xmm_c0)[0];
					}
				}
			}
		}
	};

	template<>
	struct block_mul_rm_rm<double, cpu_sse3 | cpu_fma>
	{
		// C(2xn) += A(2x2) * B(2xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[2] = { a };
			const double *ptr_b[2] = { b };
			double *ptr_c[2] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 2; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					xmm_a0 = _mm_set1_pd(a[0]);
					xmm_a1 = _mm_set1_pd(a[1]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b[0] + j);
						xmm_b1 = _mm_loadu_pd(ptr_b[1] + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_c0 = _mm_fmadd_pd(xmm_a1, xmm_b1, xmm_c0);
						// store data into memory
						_mm_storeu_pd(c, _mm_add_pd(_mm_loadu_pd(c), xmm_c0));
						c += 2;
					}
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_pd(ptr_a[0]);
				xmm_a1 = _mm_loadu_pd(ptr_a[1]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_pd(ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b0);
					xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<double*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<double*>(&xmm_c0)[1];
				}
			}
		}
	};

	template<>
	struct block_mul_rm_rm<float, cpu_avx>
	{
		// C(8xn) += A(8x8) * B(8xn)
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[8] = { a };
			const float *ptr_b[8] = { b };
			float *ptr_c[8] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_a[4] = ptr_a[3] + rsa;
			ptr_a[5] = ptr_a[4] + rsa;
			ptr_a[6] = ptr_a[5] + rsa;
			ptr_a[7] = ptr_a[6] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_b[4] = ptr_b[3] + rsb;
			ptr_b[5] = ptr_b[4] + rsb;
			ptr_b[6] = ptr_b[5] + rsb;
			ptr_b[7] = ptr_b[6] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			ptr_c[4] = ptr_c[3] + rsc;
			ptr_c[5] = ptr_c[4] + rsc;
			ptr_c[6] = ptr_c[5] + rsc;
			ptr_c[7] = ptr_c[6] + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 8; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_ps(a[0]);
					ymm_a1 = _mm256_set1_ps(a[1]);
					ymm_a2 = _mm256_set1_ps(a[2]);
					ymm_a3 = _mm256_set1_ps(a[3]);
					ymm_a4 = _mm256_set1_ps(a[4]);
					ymm_a5 = _mm256_set1_ps(a[5]);
					ymm_a6 = _mm256_set1_ps(a[6]);
					ymm_a7 = _mm256_set1_ps(a[7]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b[3] + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b[4] + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b[5] + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b[6] + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b[7] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b4);
						ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b5);
						ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b6);
						ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b7);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
						ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c5);
						ymm_c6 = _mm256_add_ps(ymm_c6, ymm_c7);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
						ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c6);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c4);
						// store data into memory
						_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
						c += 8;
					}
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_ps(ptr_a[0]);
				ymm_a1 = _mm256_loadu_ps(ptr_a[1]);
				ymm_a2 = _mm256_loadu_ps(ptr_a[2]);
				ymm_a3 = _mm256_loadu_ps(ptr_a[3]);
				ymm_a4 = _mm256_loadu_ps(ptr_a[4]);
				ymm_a5 = _mm256_loadu_ps(ptr_a[5]);
				ymm_a6 = _mm256_loadu_ps(ptr_a[6]);
				ymm_a7 = _mm256_loadu_ps(ptr_a[7]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_ps(
						ptr_b[7][j], ptr_b[6][j], ptr_b[5][j], ptr_b[4][j],
						ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b0);
					ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b0);
					ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b0);
					ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b0);
					ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b0);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c3);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c5);
					ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c7);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c2);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c6);
					ymm_a0 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a1 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_ps(ymm_a0, ymm_a1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
					ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
					ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
					ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
					ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
				}
			}
		}
		// C(8xn) += A(8xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[8] = { a };
			const float *ptr_b[8] = { b };
			float *ptr_c[8] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_a[4] = ptr_a[3] + rsa;
			ptr_a[5] = ptr_a[4] + rsa;
			ptr_a[6] = ptr_a[5] + rsa;
			ptr_a[7] = ptr_a[6] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_b[4] = ptr_b[3] + rsb;
			ptr_b[5] = ptr_b[4] + rsb;
			ptr_b[6] = ptr_b[5] + rsb;
			ptr_b[7] = ptr_b[6] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			ptr_c[4] = ptr_c[3] + rsc;
			ptr_c[5] = ptr_c[4] + rsc;
			ptr_c[6] = ptr_c[5] + rsc;
			ptr_c[7] = ptr_c[6] + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			if (aligned_n > 0)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_ps(ptr_a[0][k]);
					ymm_a1 = _mm256_set1_ps(ptr_a[1][k]);
					ymm_a2 = _mm256_set1_ps(ptr_a[2][k]);
					ymm_a3 = _mm256_set1_ps(ptr_a[3][k]);
					ymm_a4 = _mm256_set1_ps(ptr_a[4][k]);
					ymm_a5 = _mm256_set1_ps(ptr_a[5][k]);
					ymm_a6 = _mm256_set1_ps(ptr_a[6][k]);
					ymm_a7 = _mm256_set1_ps(ptr_a[7][k]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b[3] + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b[4] + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b[5] + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b[6] + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b[7] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b4);
						ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b5);
						ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b6);
						ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b7);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
						ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c5);
						ymm_c6 = _mm256_add_ps(ymm_c6, ymm_c7);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
						ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c6);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c4);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
						ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
						ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
						ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
						ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set_ps(
						ptr_a[7][k], ptr_a[6][k], ptr_a[5][k], ptr_a[4][k],
						ptr_a[3][k], ptr_a[2][k], ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_ps(ptr_b[k][j]);
						// return the product
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
						ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
						ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
						ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
						ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
					}
				}
			}
		}
		// C(mxn) += A(mx8) * B(8xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[8] = { a };
			const float *ptr_b[8] = { b };
			float *ptr_c[8] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_a[4] = ptr_a[3] + rsa;
			ptr_a[5] = ptr_a[4] + rsa;
			ptr_a[6] = ptr_a[5] + rsa;
			ptr_a[7] = ptr_a[6] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_b[4] = ptr_b[3] + rsb;
			ptr_b[5] = ptr_b[4] + rsb;
			ptr_b[6] = ptr_b[5] + rsb;
			ptr_b[7] = ptr_b[6] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			ptr_c[4] = ptr_c[3] + rsc;
			ptr_c[5] = ptr_c[4] + rsc;
			ptr_c[6] = ptr_c[5] + rsc;
			ptr_c[7] = ptr_c[6] + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < m; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_ps(a[0]);
					ymm_a1 = _mm256_set1_ps(a[1]);
					ymm_a2 = _mm256_set1_ps(a[2]);
					ymm_a3 = _mm256_set1_ps(a[3]);
					ymm_a4 = _mm256_set1_ps(a[4]);
					ymm_a5 = _mm256_set1_ps(a[5]);
					ymm_a6 = _mm256_set1_ps(a[6]);
					ymm_a7 = _mm256_set1_ps(a[7]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b[3] + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b[4] + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b[5] + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b[6] + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b[7] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b4);
						ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b5);
						ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b6);
						ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b7);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
						ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c5);
						ymm_c6 = _mm256_add_ps(ymm_c6, ymm_c7);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
						ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c6);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c4);
						// store data into memory
						_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
						c += 8;
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t i = 0; i < m; ++i)
				{
					ymm_a0 = _mm256_loadu_ps(ptr_a[i]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set_ps(
							ptr_b[7][j], ptr_b[6][j], ptr_b[5][j], ptr_b[4][j],
							ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
						// return the product
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
						ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
						ymm_c1 = _mm256_permute2f128_ps(ymm_c0, ymm_c0, _MM_SHUFFLE(0, 2, 0, 1));
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						// store data into memory
						ptr_c[i][j] += reinterpret_cast<float*>(&ymm_c0)[0];
					}
				}
			}
		}
	};

	template<>
	struct block_mul_rm_rm<float, cpu_avx | cpu_fma>
	{
		// C(8xn) += A(8x8) * B(8xn)
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[8] = { a };
			const float *ptr_b[8] = { b };
			float *ptr_c[8] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_a[4] = ptr_a[3] + rsa;
			ptr_a[5] = ptr_a[4] + rsa;
			ptr_a[6] = ptr_a[5] + rsa;
			ptr_a[7] = ptr_a[6] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_b[4] = ptr_b[3] + rsb;
			ptr_b[5] = ptr_b[4] + rsb;
			ptr_b[6] = ptr_b[5] + rsb;
			ptr_b[7] = ptr_b[6] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			ptr_c[4] = ptr_c[3] + rsc;
			ptr_c[5] = ptr_c[4] + rsc;
			ptr_c[6] = ptr_c[5] + rsc;
			ptr_c[7] = ptr_c[6] + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 8; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_ps(a[0]);
					ymm_a1 = _mm256_set1_ps(a[1]);
					ymm_a2 = _mm256_set1_ps(a[2]);
					ymm_a3 = _mm256_set1_ps(a[3]);
					ymm_a4 = _mm256_set1_ps(a[4]);
					ymm_a5 = _mm256_set1_ps(a[5]);
					ymm_a6 = _mm256_set1_ps(a[6]);
					ymm_a7 = _mm256_set1_ps(a[7]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b[3] + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b[4] + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b[5] + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b[6] + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b[7] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_c0 = _mm256_fmadd_ps(ymm_a4, ymm_b4, ymm_c0);
						ymm_c1 = _mm256_fmadd_ps(ymm_a5, ymm_b5, ymm_c1);
						ymm_c2 = _mm256_fmadd_ps(ymm_a6, ymm_b6, ymm_c2);
						ymm_c3 = _mm256_fmadd_ps(ymm_a7, ymm_b7, ymm_c3);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
						// store data into memory
						_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
						c += 8;
					}
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_ps(ptr_a[0]);
				ymm_a1 = _mm256_loadu_ps(ptr_a[1]);
				ymm_a2 = _mm256_loadu_ps(ptr_a[2]);
				ymm_a3 = _mm256_loadu_ps(ptr_a[3]);
				ymm_a4 = _mm256_loadu_ps(ptr_a[4]);
				ymm_a5 = _mm256_loadu_ps(ptr_a[5]);
				ymm_a6 = _mm256_loadu_ps(ptr_a[6]);
				ymm_a7 = _mm256_loadu_ps(ptr_a[7]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_ps(
						ptr_b[7][j], ptr_b[6][j], ptr_b[5][j], ptr_b[4][j],
						ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b0);
					ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b0);
					ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b0);
					ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b0);
					ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b0);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c3);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c5);
					ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c7);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c2);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c6);
					ymm_a0 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a1 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_ps(ymm_a0, ymm_a1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
					ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
					ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
					ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
					ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
				}
			}
		}
		// C(8xn) += A(8xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[8] = { a };
			const float *ptr_b[8] = { b };
			float *ptr_c[8] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_a[4] = ptr_a[3] + rsa;
			ptr_a[5] = ptr_a[4] + rsa;
			ptr_a[6] = ptr_a[5] + rsa;
			ptr_a[7] = ptr_a[6] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_b[4] = ptr_b[3] + rsb;
			ptr_b[5] = ptr_b[4] + rsb;
			ptr_b[6] = ptr_b[5] + rsb;
			ptr_b[7] = ptr_b[6] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			ptr_c[4] = ptr_c[3] + rsc;
			ptr_c[5] = ptr_c[4] + rsc;
			ptr_c[6] = ptr_c[5] + rsc;
			ptr_c[7] = ptr_c[6] + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_ps(ptr_a[0][k]);
					ymm_a1 = _mm256_set1_ps(ptr_a[1][k]);
					ymm_a2 = _mm256_set1_ps(ptr_a[2][k]);
					ymm_a3 = _mm256_set1_ps(ptr_a[3][k]);
					ymm_a4 = _mm256_set1_ps(ptr_a[4][k]);
					ymm_a5 = _mm256_set1_ps(ptr_a[5][k]);
					ymm_a6 = _mm256_set1_ps(ptr_a[6][k]);
					ymm_a7 = _mm256_set1_ps(ptr_a[7][k]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b[3] + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b[4] + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b[5] + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b[6] + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b[7] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_c0 = _mm256_fmadd_ps(ymm_a4, ymm_b4, ymm_c0);
						ymm_c1 = _mm256_fmadd_ps(ymm_a5, ymm_b5, ymm_c1);
						ymm_c2 = _mm256_fmadd_ps(ymm_a6, ymm_b6, ymm_c2);
						ymm_c3 = _mm256_fmadd_ps(ymm_a7, ymm_b7, ymm_c3);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
						ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
						ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
						ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
						ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set_ps(
						ptr_a[7][k], ptr_a[6][k], ptr_a[5][k], ptr_a[4][k],
						ptr_a[3][k], ptr_a[2][k], ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_ps(ptr_b[k][j]);
						// return the product
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
						ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
						ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
						ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
						ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
					}
				}
			}
		}
		// C(mxn) += A(mx8) * B(8xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_a[8] = { a };
			const float *ptr_b[8] = { b };
			float *ptr_c[8] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_a[4] = ptr_a[3] + rsa;
			ptr_a[5] = ptr_a[4] + rsa;
			ptr_a[6] = ptr_a[5] + rsa;
			ptr_a[7] = ptr_a[6] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_b[4] = ptr_b[3] + rsb;
			ptr_b[5] = ptr_b[4] + rsb;
			ptr_b[6] = ptr_b[5] + rsb;
			ptr_b[7] = ptr_b[6] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			ptr_c[4] = ptr_c[3] + rsc;
			ptr_c[5] = ptr_c[4] + rsc;
			ptr_c[6] = ptr_c[5] + rsc;
			ptr_c[7] = ptr_c[6] + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < m; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_ps(a[0]);
					ymm_a1 = _mm256_set1_ps(a[1]);
					ymm_a2 = _mm256_set1_ps(a[2]);
					ymm_a3 = _mm256_set1_ps(a[3]);
					ymm_a4 = _mm256_set1_ps(a[4]);
					ymm_a5 = _mm256_set1_ps(a[5]);
					ymm_a6 = _mm256_set1_ps(a[6]);
					ymm_a7 = _mm256_set1_ps(a[7]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b[3] + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b[4] + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b[5] + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b[6] + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b[7] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_c0 = _mm256_fmadd_ps(ymm_a4, ymm_b4, ymm_c0);
						ymm_c1 = _mm256_fmadd_ps(ymm_a5, ymm_b5, ymm_c1);
						ymm_c2 = _mm256_fmadd_ps(ymm_a6, ymm_b6, ymm_c2);
						ymm_c3 = _mm256_fmadd_ps(ymm_a7, ymm_b7, ymm_c3);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
						// store data into memory
						_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
						c += 8;
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t i = 0; i < m; ++i)
				{
					ymm_a0 = _mm256_loadu_ps(ptr_a[i]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set_ps(
							ptr_b[7][j], ptr_b[6][j], ptr_b[5][j], ptr_b[4][j],
							ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
						// return the product
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
						ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
						ymm_c1 = _mm256_permute2f128_ps(ymm_c0, ymm_c0, _MM_SHUFFLE(0, 2, 0, 1));
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
						// store data into memory
						ptr_c[i][j] += reinterpret_cast<float*>(&ymm_c0)[0];
					}
				}
			}
		}
	};

	template<>
	struct block_mul_rm_rm<double, cpu_avx>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[4] = { a };
			const double *ptr_b[4] = { b };
			double *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_pd(a[0]);
					ymm_a1 = _mm256_set1_pd(a[1]);
					ymm_a2 = _mm256_set1_pd(a[2]);
					ymm_a3 = _mm256_set1_pd(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_pd(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_pd(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_pd(ptr_b[3] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b3);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_pd(ymm_c2, ymm_c3);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c2);
						// store data into memory
						_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_pd(ptr_a[0]);
				ymm_a1 = _mm256_loadu_pd(ptr_a[1]);
				ymm_a2 = _mm256_loadu_pd(ptr_a[2]);
				ymm_a3 = _mm256_loadu_pd(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_pd(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b0);
					ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c3);
					ymm_a0 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a1 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_pd(ymm_a0, ymm_a1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
				}
			}
		}
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[4] = { a };
			const double *ptr_b[4] = { b };
			double *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_pd(ptr_a[0][k]);
					ymm_a1 = _mm256_set1_pd(ptr_a[1][k]);
					ymm_a2 = _mm256_set1_pd(ptr_a[2][k]);
					ymm_a3 = _mm256_set1_pd(ptr_a[3][k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_pd(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_pd(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_pd(ptr_b[3] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b3);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_pd(ymm_c2, ymm_c3);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c2);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set_pd(ptr_a[3][k], ptr_a[2][k], ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_pd(ptr_b[k][j]);
						// return the product
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
					}
				}
			}
		}
		// C(mxn) += A(mx4) * B(4xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[4] = { a };
			const double *ptr_b[4] = { b };
			double *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < m; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_pd(a[0]);
					ymm_a1 = _mm256_set1_pd(a[1]);
					ymm_a2 = _mm256_set1_pd(a[2]);
					ymm_a3 = _mm256_set1_pd(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_pd(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_pd(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_pd(ptr_b[3] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
						ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b2);
						ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b3);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						ymm_c2 = _mm256_add_pd(ymm_c2, ymm_c3);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c2);
						// store data into memory
						_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t i = 0; i < m; ++i)
				{
					ymm_a0 = _mm256_loadu_pd(ptr_a[i]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set_pd(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
						// return the product
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
						ymm_c1 = _mm256_permute2f128_pd(ymm_c0, ymm_c0, _MM_SHUFFLE(0, 2, 0, 1));
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						// store data into memory
						ptr_c[i][j] += reinterpret_cast<double*>(&ymm_c0)[0];
					}
				}
			}
		}
	};

	template<>
	struct block_mul_rm_rm<double, cpu_avx | cpu_fma>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[4] = { a };
			const double *ptr_b[4] = { b };
			double *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_pd(a[0]);
					ymm_a1 = _mm256_set1_pd(a[1]);
					ymm_a2 = _mm256_set1_pd(a[2]);
					ymm_a3 = _mm256_set1_pd(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_pd(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_pd(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_pd(ptr_b[3] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
						ymm_c0 = _mm256_fmadd_pd(ymm_a2, ymm_b2, ymm_c0);
						ymm_c1 = _mm256_fmadd_pd(ymm_a3, ymm_b3, ymm_c1);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						// store data into memory
						_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_pd(ptr_a[0]);
				ymm_a1 = _mm256_loadu_pd(ptr_a[1]);
				ymm_a2 = _mm256_loadu_pd(ptr_a[2]);
				ymm_a3 = _mm256_loadu_pd(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_pd(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b0);
					ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c3);
					ymm_a0 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a1 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_pd(ymm_a0, ymm_a1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
				}
			}
		}
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t aligned_p, size_t p, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[4] = { a };
			const double *ptr_b[4] = { b };
			double *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1;

			if (aligned_n > 0)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_pd(ptr_a[0][k]);
					ymm_a1 = _mm256_set1_pd(ptr_a[1][k]);
					ymm_a2 = _mm256_set1_pd(ptr_a[2][k]);
					ymm_a3 = _mm256_set1_pd(ptr_a[3][k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_pd(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_pd(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_pd(ptr_b[3] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
						ymm_c0 = _mm256_fmadd_pd(ymm_a2, ymm_b2, ymm_c0);
						ymm_c1 = _mm256_fmadd_pd(ymm_a3, ymm_b3, ymm_c1);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					ymm_a0 = _mm256_set_pd(ptr_a[3][k], ptr_a[2][k], ptr_a[1][k], ptr_a[0][k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_pd(ptr_b[k][j]);
						// return the product
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
						ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
						ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
						ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
					}
				}
			}
		}
		// C(mxn) += A(mx4) * B(4xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_a[4] = { a };
			const double *ptr_b[4] = { b };
			double *ptr_c[4] = { c };
			ptr_a[1] = ptr_a[0] + rsa;
			ptr_a[2] = ptr_a[1] + rsa;
			ptr_a[3] = ptr_a[2] + rsa;
			ptr_b[1] = ptr_b[0] + rsb;
			ptr_b[2] = ptr_b[1] + rsb;
			ptr_b[3] = ptr_b[2] + rsb;
			ptr_c[1] = ptr_c[0] + rsc;
			ptr_c[2] = ptr_c[1] + rsc;
			ptr_c[3] = ptr_c[2] + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < m; ++i)
				{
					a = ptr_a[i];
					c = ptr_c[i];
					ymm_a0 = _mm256_set1_pd(a[0]);
					ymm_a1 = _mm256_set1_pd(a[1]);
					ymm_a2 = _mm256_set1_pd(a[2]);
					ymm_a3 = _mm256_set1_pd(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b[0] + j);
						ymm_b1 = _mm256_loadu_pd(ptr_b[1] + j);
						ymm_b2 = _mm256_loadu_pd(ptr_b[2] + j);
						ymm_b3 = _mm256_loadu_pd(ptr_b[3] + j);
						// return the weighted sum
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
						ymm_c0 = _mm256_fmadd_pd(ymm_a2, ymm_b2, ymm_c0);
						ymm_c1 = _mm256_fmadd_pd(ymm_a3, ymm_b3, ymm_c1);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						// store data into memory
						_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
						c += 4;
					}
				}
			}
			if (aligned_n < n)
			{
				for (size_t i = 0; i < m; ++i)
				{
					ymm_a0 = _mm256_loadu_pd(ptr_a[i]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set_pd(ptr_b[3][j], ptr_b[2][j], ptr_b[1][j], ptr_b[0][j]);
						// return the product
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
						ymm_c1 = _mm256_permute2f128_pd(ymm_c0, ymm_c0, _MM_SHUFFLE(0, 2, 0, 1));
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
						// store data into memory
						ptr_c[i][j] += reinterpret_cast<double*>(&ymm_c0)[0];
					}
				}
			}
		}
	};

	// Class template kernel_matrix_multiply
	template<class T, size_t block_m, size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_mul_rm_rm
	{
		// C(mxn) += A(mxp) * B(pxn)
		void operator()(size_t m, size_t p, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b;
			const size_t block_rsa = block_m * rsa;
			const size_t block_rsb = block_p * rsb;
			const size_t block_rsc = block_m * rsc;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_p = p - aligned_p;
			const size_t surplus_n = n - aligned_n;
			const struct common_mul_rm_rm<T> functor;
			const struct block_mul_rm_rm<T, inst> special_functor;

			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				ptr_b = b;
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					special_functor(aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
					ptr_b += block_rsb;
				}
				if (surplus_p > 0)
					special_functor(aligned_p, p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
				a += block_rsa;
				c += block_rsc;
			}
			if (surplus_m > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					special_functor(surplus_m, aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
					ptr_b += block_rsb;
				}
				if (surplus_p > 0)
					functor(surplus_m, surplus_p, surplus_n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
			}
		}
	};

} // namespace core

#endif
