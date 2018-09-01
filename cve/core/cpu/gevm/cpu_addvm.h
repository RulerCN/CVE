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

#ifndef __CORE_CPU_ADDVM_H__
#define __CORE_CPU_ADDVM_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../kernel/gevm/kernel_gevm_float.h"
#include "../kernel/gevm/kernel_gevm_double.h"

namespace core
{
	// The multiplication of the vector and the matrix

	template <class A, class A1, class A2>
	vector<float, A>& cpu_addvm(vector<float, A> &c, const vector<float, A1> &a, const matrix<float, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.row_size() || a.size() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevm_float<8, 8, cpu_avx | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_gevm_float<8, 8, cpu_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu_inst::is_support_sse3())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevm_float<4, 4, cpu_sse3 | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_gevm_float<4, 4, cpu_sse3>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_gevm_float<4, 4, cpu_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

	template <class A, class A1, class A2>
	vector<double, A>& cpu_addvm(vector<double, A> &c, const vector<double, A1> &a, const matrix<double, A2> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.row_size() || a.size() != b.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevm_double<4, 4, cpu_avx | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_gevm_double<4, 4, cpu_avx>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else if (cpu_inst::is_support_sse3())
		{
			if (cpu_inst::is_support_fma())
				kernel_gevm_double<2, 2, cpu_sse3 | cpu_fma>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
			else
				kernel_gevm_double<2, 2, cpu_sse3>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		}
		else
			kernel_gevm_double<4, 4, cpu_none>()(b.rows(), b.row_size(), a.data(), b.data(), b.row_size(), c.data());
		return c;
	}

} // namespace core

#endif
