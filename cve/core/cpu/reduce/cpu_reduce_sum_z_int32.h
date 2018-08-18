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

#ifndef __CORE_CPU_REDUCE_ROW_SUM_Z_INT32_H__
#define __CORE_CPU_REDUCE_ROW_SUM_Z_INT32_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/reduce/kernel_reduce_sum_int32.h"

namespace core
{
	// Computes the sum of elements across the z axis of a tensor

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_reduce_sum_z(matrix<signed int, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	// Computes the sum of elements across the z axis of a tensor

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sumt_int32<4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_reduce_sum_z(tensor<signed int, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sumt_int32<8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_reduce_sumt_int32<4, 4, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sumt_int32<4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

} // namespace core

#endif
