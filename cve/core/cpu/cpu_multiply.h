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

#ifndef __CORE_CPU_MULTIPLY_H__
#define __CORE_CPU_MULTIPLY_H__

#include "../vector.h"
#include "../matrix.h"
#include "kernel/kernel_matrix_multiply.h"
#include "kernel/kernel_transpose_matrix_multiply.h"

namespace core
{
	// Vector-matrix multiplication: C = A * B or C = A * B^T
	// Parameters:
	// 1. c - output vector.
	// 2. a - input vector.
	// 3. b - input matrix.
	// 4. transpose - whether the matrix B is transposed.

	template <class A, class A1, class A2>
	vector<float, A>& cpu_multiply(vector<float, A> &c, const vector<float, A1> &a, const matrix<float, A2> &b, bool transpose = false)
	{
		c.fill(0.0F);
		if (transpose)
			return cpu_transpose_matrix_multiply(c, a, b);
		else
			return cpu_matrix_multiply(c, a, b);
	}

	template <class A, class A1, class A2>
	vector<double, A>& cpu_multiply(vector<double, A> &c, const vector<double, A1> &a, const matrix<double, A2> &b, bool transpose = false)
	{
		c.fill(0.0);
		if (transpose)
			return cpu_transpose_matrix_multiply(c, a, b);
		else
			return cpu_matrix_multiply(c, a, b);
	}

	// Matrix-matrix multiplication: C = A * B or C = A * B^T
	// Parameters:
	// 1. c - output matrix.
	// 2. a - input matrix.
	// 3. b - input matrix.
	// 4. transpose - whether the matrix B is transposed.

	template <class A, class A1, class A2>
	matrix<float, A>& cpu_multiply(matrix<float, A> &c, const matrix<float, A1> &a, const matrix<float, A2> &b, bool transpose = false)
	{
		c.fill(0.0F);
		if (transpose)
			return cpu_transpose_matrix_multiply(c, a, b);
		else
			return cpu_matrix_multiply(c, a, b);
	}

	template <class A, class A1, class A2>
	matrix<double, A>& cpu_multiply(matrix<double , A> &c, const matrix<double, A1> &a, const matrix<double, A2> &b, bool transpose = false)
	{
		c.fill(0.0);
		if (transpose)
			return cpu_transpose_matrix_multiply(c, a, b);
		else
			return cpu_matrix_multiply(c, a, b);
	}

	// Vector-matrix multiplication: C(1xn) += A(1xp) * B(pxn)
	// Parameters:
	// 1. c - output vector.
	//        | c[1][1],c[1][2],c[1][3],бн,c[1][n] |
	// 2. a - input vector.
	//        | a[1][1],a[1][2],a[1][3],бн,a[1][p] |
	// 3. b - input matrix.
	//        | b[1][1],b[1][2],b[1][3],бн,b[1][n] |
	//        | b[2][1],b[2][2],b[2][3],бн,b[2][n] |
	//        | b[3][1],b[3][2],b[3][3],бн,b[3][n] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | b[p][1],b[p][2],b[p][3],бн,b[p][n] |

	template <class A, class A1, class A2>
	vector<float, A>& cpu_matrix_multiply(vector<float, A> &c, const vector<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.length() != b.row_size() || a.length() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<float, 8, 8, inst_avx | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_matrix_multiply<float, 8, 8, inst_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<float, 4, 4, inst_sse | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_matrix_multiply<float, 4, 4, inst_sse>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_matrix_multiply<float, 4, 4, inst_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

	template <class A, class A1, class A2>
	vector<double, A>& cpu_matrix_multiply(vector<double, A> &c, const vector<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.length() != b.row_size() || a.length() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<double, 4, 4, inst_avx | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_matrix_multiply<double, 4, 4, inst_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<double, 2, 2, inst_sse | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_matrix_multiply<double, 2, 2, inst_sse>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_matrix_multiply<double, 4, 4, inst_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

	// Vector-matrix multiplication: C(1xn) += A(1xp) * B(nxp)^T
	// Parameters:
	// 1. c - output vector.
	//        | c[1][1],c[1][2],c[1][3],бн,c[1][n] |
	// 2. a - input vector.
	//        | a[1][1],a[1][2],a[1][3],бн,a[1][p] |
	// 3. b - input matrix.
	//        | b[1][1],b[1][2],b[1][3],бн,b[1][p] |
	//        | b[2][1],b[2][2],b[2][3],бн,b[2][p] |
	//        | b[3][1],b[3][2],b[3][3],бн,b[3][p] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | b[n][1],b[n][2],b[n][3],бн,b[n][p] |

	template <class A, class A1, class A2>
	vector<float, A>& cpu_transpose_matrix_multiply(vector<float, A> &c, const vector<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.length() != b.rows() || a.length() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<float, 8, 8, inst_avx | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_transpose_matrix_multiply<float, 8, 8, inst_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<float, 4, 4, inst_sse | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_transpose_matrix_multiply<float, 4, 4, inst_sse>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_transpose_matrix_multiply<float, 4, 4, inst_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

	template <class A, class A1, class A2>
	vector<double, A>& cpu_transpose_matrix_multiply(vector<double, A> &c, const vector<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.length() != b.rows() || a.length() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<double, 4, 4, inst_avx | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_transpose_matrix_multiply<double, 4, 4, inst_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<double, 2, 2, inst_sse | inst_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_transpose_matrix_multiply<double, 2, 2, inst_sse>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_transpose_matrix_multiply<double, 4, 4, inst_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

	// Matrix-matrix multiplication: C(mxn) += A(mxp) * B(pxn)
	// Parameters:
	// 1. c - output matrix.
	//        | c[1][1],c[1][2],c[1][3],бн,c[1][n] |
	//        | c[2][1],c[2][2],c[2][3],бн,c[2][n] |
	//        | c[3][1],c[3][2],c[3][3],бн,c[3][n] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | c[m][1],c[m][2],c[m][3],бн,c[m][n] |
	// 2. a - input matrix.
	//        | a[1][1],a[1][2],a[1][3],бн,a[1][p] |
	//        | a[2][1],a[2][2],a[2][3],бн,a[2][p] |
	//        | a[3][1],a[3][2],a[3][3],бн,a[3][p] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | a[m][1],a[m][2],a[m][3],бн,a[m][p] |
	// 3. b - input matrix.
	//        | b[1][1],b[1][2],b[1][3],бн,b[1][n] |
	//        | b[2][1],b[2][2],b[2][3],бн,b[2][n] |
	//        | b[3][1],b[3][2],b[3][3],бн,b[3][n] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | b[p][1],b[p][2],b[p][3],бн,b[p][n] |

	template <class A, class A1, class A2>
	matrix<float, A>& cpu_matrix_multiply(matrix<float, A> &c, const matrix<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.rows() != a.rows() || c.row_size() != b.row_size() || a.row_size() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<float, 8, 8, inst_avx | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matrix_multiply<float, 8, 8, inst_avx>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<float, 4, 4, inst_sse | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matrix_multiply<float, 4, 4, inst_sse>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_matrix_multiply<float, 4, 4, inst_none>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

	template <class A, class A1, class A2>
	matrix<double, A>& cpu_matrix_multiply(matrix<double, A> &c, const matrix<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.rows() != a.rows() || c.row_size() != b.row_size() || a.row_size() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<double, 4, 4, inst_avx | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matrix_multiply<double, 4, 4, inst_avx>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_matrix_multiply<double, 2, 2, inst_sse | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_matrix_multiply<double, 2, 2, inst_sse>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_matrix_multiply<double, 4, 4, inst_none>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

	// Matrix-matrix multiplication: C(mxn) += A(mxp) * B(nxp)^T
	// Parameters:
	// 1. c - output matrix.
	//        | c[1][1],c[1][2],c[1][3],бн,c[1][n] |
	//        | c[2][1],c[2][2],c[2][3],бн,c[2][n] |
	//        | c[3][1],c[3][2],c[3][3],бн,c[3][n] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | c[m][1],c[m][2],c[m][3],бн,c[m][n] |
	// 2. a - input matrix.
	//        | a[1][1],a[1][2],a[1][3],бн,a[1][p] |
	//        | a[2][1],a[2][2],a[2][3],бн,a[2][p] |
	//        | a[3][1],a[3][2],a[3][3],бн,a[3][p] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | a[m][1],a[m][2],a[m][3],бн,a[m][p] |
	// 3. b - input matrix.
	//        | b[1][1],b[1][2],b[1][3],бн,b[1][p] |
	//        | b[2][1],b[2][2],b[2][3],бн,b[2][p] |
	//        | b[3][1],b[3][2],b[3][3],бн,b[3][p] |
	//        |    бн  ,   бн  ,   бн,  бн,   бн   |
	//        | b[n][1],b[n][2],b[n][3],бн,b[n][p] |

	template <class A, class A1, class A2>
	matrix<float, A>& cpu_transpose_matrix_multiply(matrix<float, A> &c, const matrix<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.rows() != a.rows() || c.row_size() != b.rows() || a.row_size() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<float, 8, 8, inst_avx | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_transpose_matrix_multiply<float, 8, 8, inst_avx>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<float, 4, 4, inst_sse | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_transpose_matrix_multiply<float, 4, 4, inst_sse>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_transpose_matrix_multiply<float, 4, 4, inst_none>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

	template <class A, class A1, class A2>
	matrix<double, A>& cpu_transpose_matrix_multiply(matrix<double, A> &c, const matrix<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.rows() != a.rows() || c.row_size() != b.rows() || a.row_size() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu::is_support_avx())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<double, 4, 4, inst_avx | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_transpose_matrix_multiply<double, 4, 4, inst_avx>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu::is_support_sse())
		{
			if (cpu::is_support_fma())
				kernel_transpose_matrix_multiply<double, 2, 2, inst_sse | inst_fma>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_transpose_matrix_multiply<double, 2, 2, inst_sse>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_transpose_matrix_multiply<double, 4, 4, inst_none>()(a.rows(), b.rows(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

} // namespace core

#endif
