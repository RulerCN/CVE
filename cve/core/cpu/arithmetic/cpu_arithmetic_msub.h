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

#ifndef __CORE_CPU_ARITHMETIC_MSUB_H__
#define __CORE_CPU_ARITHMETIC_MSUB_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/arithmetic/kernel_arithmetic_msub.h"
#include "../kernel/arithmetic/kernel_arithmetic_msub_value.h"

namespace core
{
	// Multiply-Sub of value and scalar

	template <class A>
	scalar<float, A>& cpu_msub(scalar<float, A> &c, float a, const scalar<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<double, A>& cpu_msub(scalar<double, A> &c, double a, const scalar<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Multiply-Sub of value and vector

	template <class A>
	vector<float, A>& cpu_msub(vector<float, A> &c, float a, const vector<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<double, A>& cpu_msub(vector<double, A> &c, double a, const vector<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Multiply-Sub of value and matrix

	template <class A>
	matrix<float, A>& cpu_msub(matrix<float, A> &c, float a, const matrix<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<double, A>& cpu_msub(matrix<double, A> &c, double a, const matrix<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Multiply-Sub of value and tensor

	template <class A>
	tensor<float, A>& cpu_msub(tensor<float, A> &c, float a, const tensor<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<double, A>& cpu_msub(tensor<double, A> &c, double a, const tensor<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_value<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_msub_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_msub_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Multiply-Sub of scalar and scalar

	template <class A>
	scalar<float, A>& cpu_msub(scalar<float, A> &c, const scalar<float, A> &a, const scalar<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_sse | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<double, A>& cpu_msub(scalar<double, A> &c, const scalar<double, A> &a, const scalar<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_sse2 | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Multiply-Sub of vector and vector

	template <class A>
	vector<float, A>& cpu_msub(vector<float, A> &c, const vector<float, A> &a, const vector<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_sse | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	vector<double, A>& cpu_msub(vector<double, A> &c, const vector<double, A> &a, const vector<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_sse2 | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Multiply-Sub of matrix and matrix

	template <class A>
	matrix<float, A>& cpu_msub(matrix<float, A> &c, const matrix<float, A> &a, const matrix<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_sse | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<double, A>& cpu_msub(matrix<double, A> &c, const matrix<double, A> &a, const matrix<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_sse2 | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Multiply-Sub of tensor and tensor

	template <class A>
	tensor<float, A>& cpu_msub(tensor<float, A> &c, const tensor<float, A> &a, const tensor<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<float, cpu_sse | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<double, A>& cpu_msub(tensor<double, A> &c, const tensor<double, A> &a, const tensor<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_avx | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub<double, cpu_sse2 | cpu_fma>()(c.size(), a.data(), b.data(), c.data());
			else
				kernel_msub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Multiply-Sub of vector and scalar

	template <class A>
	vector<float, A>& cpu_msub(vector<float, A> &c, const vector<float, A> &a, const scalar<float, A> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<float, cpu_avx | cpu_fma>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<float, cpu_avx>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<float, cpu_sse | cpu_fma>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<float, cpu_sse>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub_element<float, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	vector<double, A>& cpu_msub(vector<double, A> &c, const vector<double, A> &a, const scalar<double, A> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<double, cpu_avx | cpu_fma>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<double, cpu_avx>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<double, cpu_sse2 | cpu_fma>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<double, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub_element<double, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	// Multiply-Sub of matrix and vector

	template <class A>
	matrix<float, A>& cpu_msub(matrix<float, A> &c, const matrix<float, A> &a, const vector<float, A> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<float, cpu_avx | cpu_fma>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<float, cpu_avx>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<float, cpu_sse | cpu_fma>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<float, cpu_sse>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub_element<float, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<double, A>& cpu_msub(matrix<double, A> &c, const matrix<double, A> &a, const vector<double, A> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<double, cpu_avx | cpu_fma>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<double, cpu_avx>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<double, cpu_sse2 | cpu_fma>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<double, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub_element<double, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	// Multiply-Sub of tensor and matrix

	template <class A>
	tensor<float, A>& cpu_msub(tensor<float, A> &c, const tensor<float, A> &a, const matrix<float, A> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<float, cpu_avx | cpu_fma>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<float, cpu_avx>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<float, cpu_sse | cpu_fma>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<float, cpu_sse>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub_element<float, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<double, A>& cpu_msub(tensor<double, A> &c, const tensor<double, A> &a, const matrix<double, A> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<double, cpu_avx | cpu_fma>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<double, cpu_avx>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_msub_element<double, cpu_sse2 | cpu_fma>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
			else
				kernel_msub_element<double, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		}
		else
			kernel_msub_element<double, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

} // namespace core

#endif
