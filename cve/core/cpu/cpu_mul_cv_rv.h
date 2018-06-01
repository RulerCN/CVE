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

#ifndef __CORE_CPU_MUL_CV_RV_H__
#define __CORE_CPU_MUL_CV_RV_H__

#include "../vector.h"
#include "../matrix.h"
#include "kernel/kernel_mul_cv_rv.h"

namespace core
{
	// The multiplication of the column vector and the row vector
	// Parameters:
	// 1. c - output row-major order matrix.
	//        | c[1][1],c[1][2],c[1][3],...,c[1][n] |
	//        | c[2][1],c[2][2],c[2][3],...,c[2][n] |
	//        | c[3][1],c[3][2],c[3][3],...,c[3][n] |
	//        |   ...  ,  ...  ,  ...  ,...,  ...   |
	//        | c[m][1],c[m][2],c[m][3],...,c[m][n] |
	// 2. a - input column vector.
	//        | a[1],a[2],a[3],...,a[m] |
	// 3. b - input row vector.
	//        | b[1],b[2],b[3],...,b[n] |

	template <class A, class A1, class A2>
	matrix<float, A>& cpu_mul_rv_cm(matrix<float, A> &c, const vector<float, A1> &a, const vector<float, A2> &b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.rows() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_cv_rv<float, 8, cpu_avx | cpu_fma>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_mul_cv_rv<float, 8, cpu_avx>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_cv_rv<float, 4, cpu_sse | cpu_fma>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_mul_cv_rv<float, 4, cpu_sse>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else
			kernel_mul_cv_rv<float, 4, cpu_none>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		return c;
	}

	template <class A, class A1, class A2>
	matrix<double, A>& cpu_mul_rv_cm(matrix<double, A> &c, const vector<double, A1> &a, const vector<double, A2> &b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.rows() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_cv_rv<double, 4, cpu_avx | cpu_fma>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_mul_cv_rv<double, 4, cpu_avx>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_cv_rv<double, 2, cpu_sse2 | cpu_fma>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_mul_cv_rv<double, 2, cpu_sse2>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else
			kernel_mul_cv_rv<double, 4, cpu_none>()(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		return c;
	}

} // namespace core

#endif
