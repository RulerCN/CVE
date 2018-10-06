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

#ifndef __CORE_CPU_ADDVV_H__
#define __CORE_CPU_ADDVV_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/gevv/kernel_gevv_float.h"
#include "../kernel/gevv/kernel_gevv_double.h"

namespace core
{
	// The multiplication of the column vector and the row vector

	template <class A, class A1, class A2>
	matrix<float, A>& cpu_addvv(matrix<float, A> &c, const vector<float, A1> &a, const vector<float, A2> &b)
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
				kernel_gevv_float<8, cpu_avx | cpu_fma>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_gevv_float<8, cpu_avx>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_float<4, cpu_sse | cpu_fma>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_gevv_float<4, cpu_sse>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else
			kernel_gevv_float<4, cpu_none>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		return c;
	}

	template <class A, class A1, class A2>
	matrix<double, A>& cpu_addvv(matrix<double, A> &c, const vector<double, A1> &a, const vector<double, A2> &b)
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
				kernel_gevv_double<4, cpu_avx | cpu_fma>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_gevv_double<4, cpu_avx>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_double<2, cpu_sse2 | cpu_fma>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
			else
				kernel_gevv_double<2, cpu_sse2>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		}
		else
			kernel_gevv_double<4, cpu_none>(a.size(), b.size(), a.data(), b.data(), c.data(), c.row_size());
		return c;
	}

	// The multiplication of the matrix and the matrix

	template <class A, class A1, class A2>
	tensor<float, A>& cpu_addvv(tensor<float, A> &c, const matrix<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.batch() != a.rows() || c.batch() != b.rows() || c.rows() != a.row_size() || c.row_size() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_float<8, cpu_avx | cpu_fma>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_gevv_float<8, cpu_avx>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_float<4, cpu_sse | cpu_fma>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_gevv_float<4, cpu_sse>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_gevv_float<4, cpu_none>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

	template <class A, class A1, class A2>
	tensor<double, A>& cpu_addvv(tensor<double, A> &c, const matrix<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.batch() != a.rows() || c.batch() != b.rows() || c.rows() != a.row_size() || c.row_size() != b.row_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_double<4, cpu_avx | cpu_fma>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_gevv_double<4, cpu_avx>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_double<2, cpu_sse2 | cpu_fma>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
			else
				kernel_gevv_double<2, cpu_sse2>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		}
		else
			kernel_gevv_double<4, cpu_none>(a.rows(), a.row_size(), b.row_size(), a.data(), a.row_size(), b.data(), b.row_size(), c.data(), c.row_size());
		return c;
	}

	// The multiplication of the tensor and the tensor

	template <class A, class A1, class A2>
	tensor<float, A>& cpu_addvv(tensor<float, A> &c, const tensor<float, A1> &a, const tensor<float, A2> &b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.batch() != a.batch() || c.batch() != b.batch() || c.rows() != a.matrix_size() || c.row_size() != b.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_float<8, cpu_avx | cpu_fma>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
			else
				kernel_gevv_float<8, cpu_avx>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_float<4, cpu_sse | cpu_fma>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
			else
				kernel_gevv_float<4, cpu_sse>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
		}
		else
			kernel_gevv_float<4, cpu_none>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
		return c;
	}

	template <class A, class A1, class A2>
	tensor<double, A>& cpu_addvv(tensor<double, A> &c, const tensor<double, A1> &a, const tensor<double, A2> &b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.batch() != a.batch() || c.batch() != b.batch() || c.rows() != a.matrix_size() || c.row_size() != b.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_double<4, cpu_avx | cpu_fma>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
			else
				kernel_gevv_double<4, cpu_avx>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevv_double<2, cpu_sse2 | cpu_fma>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
			else
				kernel_gevv_double<2, cpu_sse2>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
		}
		else
			kernel_gevv_double<4, cpu_none>(a.batch(), a.matrix_size(), b.matrix_size(), a.data(), a.matrix_size(), b.data(), b.matrix_size(), c.data(), c.row_size());
		return c;
	}

} // namespace core

#endif
