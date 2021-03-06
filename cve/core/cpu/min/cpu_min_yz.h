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

#ifndef __CORE_CPU_MIN_YZ_H__
#define __CORE_CPU_MIN_YZ_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/min/kernel_mint.h"

namespace core
{
	// Computes the min of elements across the y and z axis of a tensor

	template <class A>
	matrix<signed char, A>& cpu_min_yz(matrix<signed char, A> &b, const tensor<signed char, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<signed char, 32, 32, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<signed char, 16, 16, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<signed char, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned char, A>& cpu_min_yz(matrix<unsigned char, A> &b, const tensor<unsigned char, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<unsigned char, 32, 32, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<unsigned char, 16, 16, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<unsigned char, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<signed short, A>& cpu_min_yz(matrix<signed short, A> &b, const tensor<signed short, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<signed short, 16, 16, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<signed short, 8, 8, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<signed short, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned short, A>& cpu_min_yz(matrix<unsigned short, A> &b, const tensor<unsigned short, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<unsigned short, 16, 16, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<unsigned short, 8, 8, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<unsigned short, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<signed int, A>& cpu_min_yz(matrix<signed int, A> &b, const tensor<signed int, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<signed int, 8, 8, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<signed int, 4, 4, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<signed int, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned int, A>& cpu_min_yz(matrix<unsigned int, A> &b, const tensor<unsigned int, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<unsigned int, 8, 8, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<unsigned int, 4, 4, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<unsigned int, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<float, A>& cpu_min_yz(matrix<float, A> &b, const tensor<float, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_max);
		if (cpu_inst::is_support_avx())
			kernel_mint<float, 8, 8, cpu_avx>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_mint<float, 4, 4, cpu_sse>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<float, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<double, A>& cpu_min_yz(matrix<double, A> &b, const tensor<double, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_max);
		if (cpu_inst::is_support_avx())
			kernel_mint<double, 4, 4, cpu_avx>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_mint<double, 2, 2, cpu_sse2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<double, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the min of elements across the y and z axis of a tensor

	template <class A>
	tensor<signed char, A>& cpu_min_yz(tensor<signed char, A> &b, const tensor<signed char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<signed char, 32, 32, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<signed char, 16, 16, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<signed char, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned char, A>& cpu_min_yz(tensor<unsigned char, A> &b, const tensor<unsigned char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<unsigned char, 32, 32, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<unsigned char, 16, 16, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<unsigned char, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<signed short, A>& cpu_min_yz(tensor<signed short, A> &b, const tensor<signed short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<signed short, 16, 16, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<signed short, 8, 8, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<signed short, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned short, A>& cpu_min_yz(tensor<unsigned short, A> &b, const tensor<unsigned short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<unsigned short, 16, 16, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<unsigned short, 8, 8, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<unsigned short, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<signed int, A>& cpu_min_yz(tensor<signed int, A> &b, const tensor<signed int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<signed int, 8, 8, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<signed int, 4, 4, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<signed int, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned int, A>& cpu_min_yz(tensor<unsigned int, A> &b, const tensor<unsigned int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_max);
		if (cpu_inst::is_support_avx2())
			kernel_mint<unsigned int, 8, 8, cpu_avx2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_mint<unsigned int, 4, 4, cpu_sse41>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<unsigned int, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<float, A>& cpu_min_yz(tensor<float, A> &b, const tensor<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_max);
		if (cpu_inst::is_support_avx())
			kernel_mint<float, 8, 8, cpu_avx>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_mint<float, 4, 4, cpu_sse>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<float, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<double, A>& cpu_min_yz(tensor<double, A> &b, const tensor<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_max);
		if (cpu_inst::is_support_avx())
			kernel_mint<double, 4, 4, cpu_avx>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_mint<double, 2, 2, cpu_sse2>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_mint<double, 4, 4, cpu_none>(a.batch() * a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

} // namespace core

#endif
