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

#ifndef __CORE_CPU_REDUCE_ROW_SUM_H__
#define __CORE_CPU_REDUCE_ROW_SUM_H__

#include "../vector.h"
#include "../matrix.h"
#include "kernel/kernel_reduce_row_sum.h"

namespace core
{
	// Get the summation of each row of matrix: b[i] += a[i][j]
	//----------------------------------------------------------------
	// input:
	//              | a[1][1],a[1][2],a[1][3],бн,a[1][n] |
	//              | a[2][1],a[2][2],a[2][3],бн,a[2][n] |
	//          A = | a[3][1],a[3][2],a[3][3],бн,a[3][n] |
	//              |    бн  ,   бн  ,   бн,  бн,   бн   |
	//              | a[m][1],a[m][2],a[m][3],бн,a[m][n] |
	// output:
	//          B = | b[1],b[2],b[3],бн,b[m] |
	//----------------------------------------------------------------

	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_reduce_row_sum(vector<T1, A1> &b, const matrix<T2, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		kernel_reduce_row_sum<T2, T1, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce_row_sum(vector<signed int, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<signed char, signed int, 16, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<signed char, signed int, 8, 16, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<signed char, signed int, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce_row_sum(vector<float, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<signed char, float, 16, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<signed char, float, 8, 16, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<signed char, float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce_row_sum(vector<signed int, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<unsigned char, signed int, 16, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<unsigned char, signed int, 8, 16, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<unsigned char, signed int, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce_row_sum(vector<float, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<unsigned char, float, 16, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<unsigned char, float, 8, 16, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<unsigned char, float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce_row_sum(vector<signed int, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<signed short, signed int, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<signed short, signed int, 4, 8, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<signed short, signed int, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce_row_sum(vector<float, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<signed short, float, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<signed short, float, 4, 8, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<signed short, float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce_row_sum(vector<signed int, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<unsigned short, signed int, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<unsigned short, signed int, 4, 8, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<unsigned short, signed int, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce_row_sum(vector<float, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<unsigned short, float, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse41())
			kernel_reduce_row_sum<unsigned short, float, 4, 8, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<unsigned short, float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce_row_sum(vector<signed int, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<signed int, signed int, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_ssse3())
			kernel_reduce_row_sum<signed int, signed int, 4, 4, cpu_ssse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<signed int, signed int, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce_row_sum(vector<float, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<signed int, float, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_ssse3())
			kernel_reduce_row_sum<signed int, float, 4, 4, cpu_ssse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<signed int, float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce_row_sum(vector<float, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx2())
			kernel_reduce_row_sum<unsigned int, float, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse3())
			kernel_reduce_row_sum<unsigned int, float, 4, 4, cpu_sse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<unsigned int, float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce_row_sum(vector<float, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx())
			kernel_reduce_row_sum<float, float, 8, 8, cpu_avx>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse3())
			kernel_reduce_row_sum<float, float, 4, 4, cpu_sse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<float, float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(vector_invalid_size);

		if (cpu::is_support_avx())
			kernel_reduce_row_sum<double, double, 4, 4, cpu_avx>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu::is_support_sse3())
			kernel_reduce_row_sum<double, double, 2, 2, cpu_sse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_row_sum<double, double, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

} // namespace core

#endif
