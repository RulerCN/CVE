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

#ifndef __CORE_CPU_MULADD_H__
#define __CORE_CPU_MULADD_H__

#include "../scalar.h"
#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include "kernel/kernel_muladd.h"

namespace core
{
	// The muladd function for scalar

	template <class A1, class A2>
	scalar<float, A2>& cpu_muladd(const float a, const scalar<float, A1> &b, const scalar<float, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<double, A2>& cpu_muladd(const double a, const scalar<double, A1> &b, const scalar<double, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// The muladd function for vector

	template <class A1, class A2>
	vector<float, A2>& cpu_muladd(const float a, const vector<float, A1> &b, const vector<float, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<double, A2>& cpu_muladd(const double a, const vector<double, A1> &b, const vector<double, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// The muladd function for matrix

	template <class A1, class A2>
	matrix<float, A2>& cpu_muladd(const float a, const matrix<float, A1> &b, matrix<float, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<double, A2>& cpu_muladd(const double a, const matrix<double, A1> &b, matrix<double, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// The muladd function for tensor

	template <class A1, class A2>
	tensor<float, A2>& cpu_muladd(const float a, const tensor<float, A1> &b, tensor<float, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<float, cpu_sse | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<double, A2>& cpu_muladd(const double a, const tensor<double, A1> &b, tensor<double, A2> &c)
	{
		if (c.empty() || b.empty()
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_avx | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse2())
		{
			if (cpu_inst::is_support_fma())
				kernel_muladd<double, cpu_sse2 | cpu_fma>()(c.size(), a, b.data(), c.data());
			else
				kernel_muladd<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		}
		else
			kernel_muladd<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

} // namespace core

#endif
