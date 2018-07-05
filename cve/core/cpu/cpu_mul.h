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

#ifndef __CORE_CPU_MUL_H__
#define __CORE_CPU_MUL_H__

#include "../scalar.h"
#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include "kernel/kernel_mul.h"

namespace core
{
	// The mul function for scalar

	template <class A1, class A2>
	scalar<float, A1>& cpu_mul(scalar<float, A1> &b, const scalar<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_mul<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_mul<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<double, A1>& cpu_mul(scalar<double, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_mul<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_mul<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The mul function for vector

	template <class A1, class A2>
	vector<float, A1>& cpu_mul(vector<float, A1> &b, const vector<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_mul<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_mul<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_mul(vector<double, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_mul<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_mul<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The mul function for matrix

	template <class A1, class A2>
	matrix<float, A1>& cpu_mul(matrix<float, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_mul<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_mul<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_mul(matrix<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_mul<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_mul<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The mul function for tensor

	template <class A1, class A2>
	tensor<float, A1>& cpu_mul(tensor<float, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_mul<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_mul<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_mul(tensor<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_mul<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_mul<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_mul<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

} // namespace core

#endif
