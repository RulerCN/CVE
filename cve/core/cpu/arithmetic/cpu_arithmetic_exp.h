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

#ifndef __CORE_CPU_ARITHMETIC_EXP_H__
#define __CORE_CPU_ARITHMETIC_EXP_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/arithmetic/kernel_arithmetic_exp.h"

namespace core
{
	// The exponential function for scalar
	template <class T, class A1, class A2>
	scalar<T, A1>& cpu_exp(scalar<T, A1> &b, const scalar<T, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_exp<T, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The exponential function for vector
	template <class T, class A1, class A2>
	vector<T, A1>& cpu_exp(vector<T, A1> &b, const vector<T, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_exp<T, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The exponential function for matrix
	template <class T, class A1, class A2>
	matrix<T, A1>& cpu_exp(matrix<T, A1> &b, const matrix<T, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_exp<T, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The exponential function for tensor
	template <class T, class A1, class A2>
	tensor<T, A1>& cpu_exp(tensor<T, A1> &b, const tensor<T, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_avx2 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_avx2>()(b.size(), a.data(), b.data());
		}
		else if (cpu_inst::is_support_sse41())
		{
			if (cpu_inst::is_support_fma())
				kernel_exp<T, cpu_sse41 | cpu_fma>()(b.size(), a.data(), b.data());
			else
				kernel_exp<T, cpu_sse41>()(b.size(), a.data(), b.data());
		}
		else
			kernel_exp<T, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

} // namespace core

#endif
