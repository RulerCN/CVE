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

#ifndef __CORE_CPU_MUL_RV_CM_H__
#define __CORE_CPU_MUL_RV_CM_H__

#include "../vector.h"
#include "../matrix.h"
#include "kernel/kernel_mul_rv_cm.h"

namespace core
{
	// The multiplication of the row vector and the column-major order matrix
	// Parameters:
	// 1. c - output row vector.
	//        | c[1],c[2],c[3],бн,c[n] |
	// 2. a - input row vector.
	//        | a[1],a[2],a[3],бн,a[p] |
	// 3. b - input column-major order matrix.
	//        | b[1][1],b[1][2],b[1][3],бн,b[1][p] |
	//        | b[2][1],b[2][2],b[2][3],бн,b[2][p] |
	//        | b[3][1],b[3][2],b[3][3],бн,b[3][p] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | b[n][1],b[n][2],b[n][3],бн,b[n][p] |

	template <class A, class A1, class A2>
	vector<float, A>& cpu_mul_rv_cm(vector<float, A> &c, const vector<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.rows() || a.size() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_rv_cm<float, 8, 8, cpu_avx | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_mul_rv_cm<float, 8, 8, cpu_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_rv_cm<float, 4, 4, cpu_sse | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_mul_rv_cm<float, 4, 4, cpu_sse>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_mul_rv_cm<float, 4, 4, cpu_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

	template <class A, class A1, class A2>
	vector<double, A>& cpu_mul_rv_cm(vector<double, A> &c, const vector<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.rows() || a.size() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_rv_cm<double, 4, 4, cpu_avx | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_mul_rv_cm<double, 4, 4, cpu_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_mul_rv_cm<double, 2, 2, cpu_sse2 | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_mul_rv_cm<double, 2, 2, cpu_sse2>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_mul_rv_cm<double, 4, 4, cpu_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

} // namespace core

#endif
