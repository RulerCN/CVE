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

#ifndef __CORE_CPU_MATMUL_H__
#define __CORE_CPU_MATMUL_H__

#include "matmul/cpu_matmul_cvrv.h"
#include "matmul/cpu_matmul_rmcv.h"
#include "matmul/cpu_matmul_rvrm.h"
#include "matmul/cpu_matmul_rvcm.h"
#include "matmul/cpu_matmul_rmrm.h"
#include "matmul/cpu_matmul_rmcm.h"

namespace core
{
	// The multiplication of the column vector and the row vector
	// D = A * B
	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_matmul(matrix<T, A> &d, const vector<T, A1> &a, const vector<T, A2> &b)
	{
		d.fill(0);
		return cpu_matmul_cvrv(d, a, b);
	}

	// The multiplication of the column vector and the row vector
	// D = A * B + C
	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_matmul(matrix<T, A> &d, const vector<T, A1> &a, const vector<T, A2> &b, const matrix<T, A> &c)
	{
		d.fill(c);
		return cpu_matmul_cvrv(d, a, b);
	}

	// The multiplication of the matrix and the column vector
	// D = A * B
	template <class T, class A, class A1, class A2>
	vector<T, A>& cpu_matmul(vector<T, A> &d, const matrix<T, A1> &a, const vector<T, A2> &b)
	{
		d.fill(0);
		return cpu_matmul_rmcv(d, a, b);
	}

	// The multiplication of the matrix and the column vector
	// D = A * B + C
	template <class T, class A, class A1, class A2>
	vector<T, A>& cpu_matmul(vector<T, A> &d, const matrix<T, A1> &a, const vector<T, A> &c, const vector<T, A2> &b)
	{
		d.fill(c);
		return cpu_matmul_rmcv(d, a, b);
	}

	// The multiplication of the row vector and the matrix
	// D = A * B
	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_matmul(matrix<T, A> &d, const vector<T, A1> &a, const matrix<T, A2> &b, bool row_major = true)
	{
		d.fill(0);
		if (row_major)
			return cpu_matmul_rvrm(d, a, b);
		else
			return cpu_matmul_rvcm(d, a, b);
	}

	// The multiplication of the row vector and the matrix
	// D = A * B + C
	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_matmul(matrix<T, A> &d, const vector<T, A1> &a, const matrix<T, A2> &b, const matrix<T, A> &c, bool row_major = true)
	{
		d.fill(c);
		if (row_major)
			return cpu_matmul_rvrm(d, a, b);
		else
			return cpu_matmul_rvcm(d, a, b);
	}

	// The multiplication of the matrix and the matrix
	// D = A * B
	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_matmul(matrix<T, A> &d, const matrix<T, A1> &a, const matrix<T, A2> &b, bool row_major = true)
	{
		d.fill(0);
		if (row_major)
			return cpu_matmul_rmrm(d, a, b);
		else
			return cpu_matmul_rmcm(d, a, b);
	}

	// The multiplication of the matrix and the matrix
	// D = A * B + C
	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_matmul(matrix<T, A> &d, const matrix<T, A1> &a, const matrix<T, A2> &b, const matrix<T, A2> &c, bool row_major = true)
	{
		d.fill(c);
		if (row_major)
			return cpu_matmul_rmrm(d, a, b);
		else
			return cpu_matmul_rmcm(d, a, b);
	}

} // namespace core

#endif
