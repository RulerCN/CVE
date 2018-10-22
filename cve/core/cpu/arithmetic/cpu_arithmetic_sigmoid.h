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

#ifndef __CORE_CPU_ARITHMETIC_SIGMOID_H__
#define __CORE_CPU_ARITHMETIC_SIGMOID_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/arithmetic/kernel_arithmetic_sigmoid.h"

namespace core
{
	// The sigmoid function for scalar

	template <class A>
	scalar<float, A>& cpu_sigmoid(scalar<float, A> &b, const scalar<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A>
	scalar<double, A>& cpu_sigmoid(scalar<double, A> &b, const scalar<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The sigmoid function for vector

	template <class A>
	vector<float, A>& cpu_sigmoid(vector<float, A> &b, const vector<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A>
	vector<double, A>& cpu_sigmoid(vector<double, A> &b, const vector<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The sigmoid function for matrix

	template <class A>
	matrix<float, A>& cpu_sigmoid(matrix<float, A> &b, const matrix<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A>
	matrix<double, A>& cpu_sigmoid(matrix<double, A> &b, const matrix<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The sigmoid function for tensor

	template <class A>
	tensor<float, A>& cpu_sigmoid(tensor<float, A> &b, const tensor<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<float, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<float, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A>
	tensor<double, A>& cpu_sigmoid(tensor<double, A> &b, const tensor<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_sigmoid<double, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_sigmoid<double, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_sigmoid<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

} // namespace core

#endif
