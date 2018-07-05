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

#ifndef __CORE_CPU_MATMUL_RMRM_H__
#define __CORE_CPU_MATMUL_RMRM_H__

#include "../matrix.h"
#include "kernel/kernel_matmul_rmrm.h"

namespace core
{
	// The multiplication of the row-major order matrix and the row-major order matrix
	// Parameters:
	// 1. c - output row-major order matrix.
	//        | c[1][1],c[1][2],c[1][3],...,c[1][n] |
	//        | c[2][1],c[2][2],c[2][3],...,c[2][n] |
	//        | c[3][1],c[3][2],c[3][3],...,c[3][n] |
	//        |   ...  ,  ...  ,  ...  ,...,  ...   |
	//        | c[m][1],c[m][2],c[m][3],...,c[m][n] |
	// 2. a - input row-major order matrix.
	//        | a[1][1],a[1][2],a[1][3],...,a[1][p] |
	//        | a[2][1],a[2][2],a[2][3],...,a[2][p] |
	//        | a[3][1],a[3][2],a[3][3],...,a[3][p] |
	//        |   ...  ,  ...  ,  ...  ,...,  ...   |
	//        | a[m][1],a[m][2],a[m][3],...,a[m][p] |
	// 3. b - input row-major order matrix.
	//        | b[1][1],b[1][2],b[1][3],...,b[1][n] |
	//        | b[2][1],b[2][2],b[2][3],...,b[2][n] |
	//        | b[3][1],b[3][2],b[3][3],...,b[3][n] |
	//        |   ...  ,  ...  ,  ...  ,...,  ...   |
	//        | b[p][1],b[p][2],b[p][3],...,b[p][n] |

	template <class A, class A1, class A2>
	matrix<float, A>& cpu_matmul_rmrm(matrix<float, A> &c, const matrix<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.rows() != a.rows() || c.row_size() != b.row_size() || a.row_size() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_matmul_rmrm<float, 8, 8, 8, cpu_avx | cpu_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matmul_rmrm<float, 8, 8, 8, cpu_avx>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse3())
		{
			if (cpu_inst::is_support_fma())
				kernel_matmul_rmrm<float, 4, 4, 4, cpu_sse3 | cpu_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matmul_rmrm<float, 4, 4, 4, cpu_sse3>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_matmul_rmrm<float, 4, 4, 4, cpu_none>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

	template <class A, class A1, class A2>
	matrix<double, A>& cpu_matmul_rmrm(matrix<double, A> &c, const matrix<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.rows() != a.rows() || c.row_size() != b.row_size() || a.row_size() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_matmul_rmrm<double, 4, 4, 4, cpu_avx | cpu_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matmul_rmrm<double, 4, 4, 4, cpu_avx>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_ss3e())
		{
			if (cpu_inst::is_support_fma())
				kernel_matmul_rmrm<double, 2, 2, 2, cpu_sse3 | cpu_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matmul_rmrm<double, 2, 2, 2, cpu_sse3>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_matmul_rmrm<double, 4, 4, 4, cpu_none>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

} // namespace core

#endif
