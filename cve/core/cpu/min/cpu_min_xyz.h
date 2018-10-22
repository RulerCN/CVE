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

#ifndef __CORE_CPU_MIN_XYZ_H__
#define __CORE_CPU_MIN_XYZ_H__

#include "../../tensor.h"
#include "../kernel/min/kernel_min.h"

namespace core
{
	// Computes the min of elements of a tensor

	template <class A>
	signed char& cpu_min_xyz(signed char &b, const tensor<signed char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = int8_max;
		if (cpu_inst::is_support_avx2())
			kernel_min<signed char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_min<signed char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned char& cpu_min_xyz(unsigned char &b, const tensor<unsigned char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = uint8_max;
		if (cpu_inst::is_support_avx2())
			kernel_min<unsigned char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_min<unsigned char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	signed short& cpu_min_xyz(signed short &b, const tensor<signed short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = int16_max;
		if (cpu_inst::is_support_avx2())
			kernel_min<signed short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_min<signed short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned short& cpu_min_xyz(unsigned short &b, const tensor<unsigned short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = uint16_max;
		if (cpu_inst::is_support_avx2())
			kernel_min<unsigned short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_min<unsigned short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	signed int& cpu_min_xyz(signed int &b, const tensor<signed int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = int32_max;
		if (cpu_inst::is_support_avx2())
			kernel_min<signed int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_min<signed int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned int& cpu_min_xyz(unsigned int &b, const tensor<unsigned int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = uint32_max;
		if (cpu_inst::is_support_avx2())
			kernel_min<unsigned int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_min<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	float& cpu_min_xyz(float &b, const tensor<float, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = flt_max;
		if (cpu_inst::is_support_avx())
			kernel_min<float, 8, 8, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse())
			kernel_min<float, 4, 4, cpu_sse>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_min_xyz(double &b, const tensor<double, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		b = dbl_max;
		if (cpu_inst::is_support_avx())
			kernel_min<double, 4, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse2())
			kernel_min<double, 2, 2, cpu_sse2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	// Computes the min of elements of a tensor

	template <class A>
	tensor<signed char, A>& cpu_min_xyz(tensor<signed char, A> &b, const tensor<signed char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_max);
		if (cpu_inst::is_support_avx2())
			kernel_min<signed char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_min<signed char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned char, A>& cpu_min_xyz(tensor<unsigned char, A> &b, const tensor<unsigned char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_max);
		if (cpu_inst::is_support_avx2())
			kernel_min<unsigned char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_min<unsigned char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	tensor<signed short, A>& cpu_min_xyz(tensor<signed short, A> &b, const tensor<signed short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_max);
		if (cpu_inst::is_support_avx2())
			kernel_min<signed short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_min<signed short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned short, A>& cpu_min_xyz(tensor<unsigned short, A> &b, const tensor<unsigned short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_max);
		if (cpu_inst::is_support_avx2())
			kernel_min<unsigned short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_min<unsigned short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	tensor<signed int, A>& cpu_min_xyz(tensor<signed int, A> &b, const tensor<signed int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_max);
		if (cpu_inst::is_support_avx2())
			kernel_min<signed int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_min<signed int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned int, A>& cpu_min_xyz(tensor<unsigned int, A> &b, const tensor<unsigned int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_max);
		if (cpu_inst::is_support_avx2())
			kernel_min<unsigned int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_min<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	tensor<float, A>& cpu_min_xyz(tensor<float, A> &b, const tensor<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_max);
		if (cpu_inst::is_support_avx())
			kernel_min<float, 8, 8, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_min<float, 4, 4, cpu_sse>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	tensor<double, A>& cpu_min_xyz(tensor<double, A> &b, const tensor<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_max);
		if (cpu_inst::is_support_avx())
			kernel_min<double, 4, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_min<double, 2, 2, cpu_sse2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_min<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

} // namespace core

#endif
