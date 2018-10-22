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

#ifndef __CORE_CPU_ADDVVT_H__
#define __CORE_CPU_ADDVVT_H__

#include "../../vector.h"
#include "../kernel/gevv/kernel_gevvt_float.h"
#include "../kernel/gevv/kernel_gevvt_double.h"

namespace core
{
	// The multiplication of the column vector and the row vector

	template <class A>
	float& cpu_addvvt(float &c, const vector<float, A> &a, const vector<float, A> &b)
	{
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_float<8, cpu_avx | cpu_fma>(a.size(), a.data(), b.data(), &c);
			else
				kernel_gevvt_float<8, cpu_avx>(a.size(), a.data(), b.data(), &c);
		}
		else if (cpu_inst::is_support_sse3())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_float<4, cpu_sse3 | cpu_fma>(a.size(), a.data(), b.data(), &c);
			else
				kernel_gevvt_float<4, cpu_sse3>(a.size(), a.data(), b.data(), &c);
		}
		else
			kernel_gevvt_float<4, cpu_none>(a.size(), a.data(), b.data(), &c);
		return c;
	}

	template <class A>
	double& cpu_addvvt(double &c, const vector<double, A> &a, const vector<double, A> &b)
	{
		if (a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_double<4, cpu_avx | cpu_fma>(a.size(), a.data(), b.data(), &c);
			else
				kernel_gevvt_double<4, cpu_avx>(a.size(), a.data(), b.data(), &c);
		}
		else if (cpu_inst::is_support_sse3())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_double<2, cpu_sse3 | cpu_fma>(a.size(), a.data(), b.data(), &c);
			else
				kernel_gevvt_double<2, cpu_sse3>(a.size(), a.data(), b.data(), &c);
		}
		else
			kernel_gevvt_double<4, cpu_none>(a.size(), a.data(), b.data(), &c);
		return c;
	}

	// The multiplication of the column vector and the row vector

	template <class A>
	vector<float, A>& cpu_addvvt(vector<float, A> &c, const vector<float, A> &a, const vector<float, A> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_float<8, cpu_avx | cpu_fma>(a.size(), a.data(), b.data(), c.data());
			else
				kernel_gevvt_float<8, cpu_avx>(a.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse3())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_float<4, cpu_sse3 | cpu_fma>(a.size(), a.data(), b.data(), c.data());
			else
				kernel_gevvt_float<4, cpu_sse3>(a.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_gevvt_float<4, cpu_none>(a.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A>
	vector<double, A>& cpu_addvvt(vector<double, A> &c, const vector<double, A> &a, const vector<double, A> &b)
	{
		if (c.empty() || a.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_double<4, cpu_avx | cpu_fma>(a.size(), a.data(), b.data(), c.data());
			else
				kernel_gevvt_double<4, cpu_avx>(a.size(), a.data(), b.data(), c.data());
		}
		else if (cpu_inst::is_support_sse3())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevvt_double<2, cpu_sse3 | cpu_fma>(a.size(), a.data(), b.data(), c.data());
			else
				kernel_gevvt_double<2, cpu_sse3>(a.size(), a.data(), b.data(), c.data());
		}
		else
			kernel_gevvt_double<4, cpu_none>(a.size(), a.data(), b.data(), c.data());
		return c;
	}

} // namespace core

#endif
