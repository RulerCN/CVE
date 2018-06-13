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

#ifndef __CORE_CPU_REPLICATE_H__
#define __CORE_CPU_REPLICATE_H__

#include "../vector.h"
#include "../matrix.h"
#include "kernel/kernel_replicate.h"

namespace core
{
	// Replicate and tile a vector to a matrix
	//----------------------------------------------------------------
	// 1. b - output matrix.
	//        | a[1][1],...,a[1][n],a[1][1],...,a[1][n],... |
	//        | a[2][1],...,a[2][n],a[2][1],...,a[2][n],... |
	//        | a[3][1],...,a[3][n],a[3][1],...,a[3][n],... |
	//        |   ...  ,...,  ...  ,  ...  ,...,  ...  ,... |
	// 2. a - input vector.
	//        | a[1],a[2],a[3],...,a[n] |
	//----------------------------------------------------------------

	template <class T, class A1, class A2>
	matrix<T, A1>& cpu_replicate(matrix<T, A1> &b, const vector<T, A2> &a, size_t m, size_t n)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (|| b.rows() != m || b.row_size() != a.size() * n)
			throw ::std::invalid_argument(invalid_shape);

		kernel_replicate(m, n, a.data(), a.size(), b.data(), b.row_size());
		return b;
	}

	// Replicate and tile a matrix to a matrix
	//----------------------------------------------------------------
	// 1. b - output matrix.
	//        | a[1][1],...,a[1][n],a[1][1],...,a[1][n],... |
	//        |   ...  ,...,  ...  ,  ...  ,...,  ...  ,... |
	//        | a[m][1],...,a[m][n],a[m][1],...,a[m][n],... |
	//        | a[1][1],...,a[1][n],a[1][1],...,a[1][n],... |
	//        |   ...  ,...,  ...  ,  ...  ,...,  ...  ,... |
	//        | a[m][1],...,a[m][n],a[m][1],...,a[m][n],... |
	//        |   ...  ,...,  ...  ,  ...  ,...,  ...  ,... |
	// 2. a - input vector.
	//        | a[1][1],a[1][2],a[1][3],...,a[1][n] |
	//        | a[2][1],a[2][2],a[2][3],...,a[2][n] |
	//        | a[3][1],a[3][2],a[3][3],...,a[3][n] |
	//        |   ...  ,  ...  ,  ...  ,...,  ...   |
	//        | a[m][1],a[m][2],a[m][3],...,a[m][n] |
	//----------------------------------------------------------------

	template <class T, class A1, class A2>
	matrix<T, A1>& cpu_replicate(matrix<T, A1> &b, const matrix<T, A2> &a, size_t m, size_t n)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.rows() != a.rows() * m || b.row_size() != a.row_size() * n)
			throw ::std::invalid_argument(invalid_shape);

		kernel_replicate(m, n, a.data(), a.row_size(), a.rows(), b.data(), b.row_size());
		return b;
	}

} // namespace core

#endif
