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

#ifndef __CORE_CPU_TRANSPOSE_H__
#define __CORE_CPU_TRANSPOSE_H__

#include "../matrix.h"
#include "kernel/kernel_transpose.h"

namespace core
{
	// Convert the matrix to transpose matrix: b[j][i] = a[i][j]
	//----------------------------------------------------------------
	// input:
	//              | a[1][1],a[1][2],a[1][3],бн,a[1][n] |
	//              | a[2][1],a[2][2],a[2][3],бн,a[2][n] |
	//          A = | a[3][1],a[3][2],a[3][3],бн,a[3][n] |
	//              |    бн  ,   бн  ,   бн,  бн,   бн   |
	//              | a[m][1],a[m][2],a[m][3],бн,a[m][n] |
	// output:
	//              | b[1][1],b[1][2],b[1][3],бн,b[1][m] |
	//              | b[2][1],b[2][2],b[2][3],бн,b[2][m] |
	//          B = | b[3][1],b[3][2],b[3][3],бн,b[3][m] |
	//              |    бн  ,   бн  ,   бн,  бн,   бн   |
	//              | b[n][1],b[n][2],b[n][3],бн,b[n][m] |
	//----------------------------------------------------------------

	template <class T, class A1, class A2>
	matrix<T, A1>& cpu_transpose(matrix<T, A1> &b, const matrix<T, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		kernel_transpose<T, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<signed char, A1>& cpu_transpose(matrix<signed char, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_transpose<signed char, 32, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_transpose<signed char, 16, 8, cpu_sse2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<signed char, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& cpu_transpose(matrix<unsigned char, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_transpose<unsigned char, 32, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_transpose<unsigned char, 16, 8, cpu_sse2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<unsigned char, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& cpu_transpose(matrix<signed short, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_transpose<signed short, 16, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_transpose<signed short, 8, 8, cpu_sse2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<signed short, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& cpu_transpose(matrix<unsigned short, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_transpose<unsigned short, 16, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_transpose<unsigned short, 8, 8, cpu_sse2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<unsigned short, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_transpose(matrix<signed int, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_transpose<signed int, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_transpose<signed int, 4, 4, cpu_sse2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<signed int, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& cpu_transpose(matrix<unsigned int, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_transpose<unsigned int, 8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_transpose<unsigned int, 4, 4, cpu_sse2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<unsigned int, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_transpose(matrix<float, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_transpose<float, 8, 8, cpu_avx>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse())
			kernel_transpose<float, 4, 4, cpu_sse>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<float, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_transpose(matrix<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.row_size() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_transpose<double, 4, 4, cpu_avx>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_transpose<double, 2, 2, cpu_sse2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		else
			kernel_transpose<double, 4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data(), b.row_size());
		return b;
	}

} // namespace core

#endif
