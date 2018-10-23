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

#ifndef __CORE_CPU_FILL_TRILINEAR_H__
#define __CORE_CPU_FILL_TRILINEAR_H__

#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/fill/kernel_fill_trilinear.h"

namespace core
{
	// Trilinear fill a matrix

	template <class A>
	matrix<signed char, A>& cpu_trilinear(matrix<signed char, A> &e, signed char a, signed char b, signed char c, signed char d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed char, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed char, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed char, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<unsigned char, A>& cpu_trilinear(matrix<unsigned char, A> &e, unsigned char a, unsigned char b, unsigned char c, unsigned char d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned char, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned char, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned char, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<signed short, A>& cpu_trilinear(matrix<signed short, A> &e, signed short a, signed short b, signed short c, signed short d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed short, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed short, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed short, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<unsigned short, A>& cpu_trilinear(matrix<unsigned short, A> &e, unsigned short a, unsigned short b, unsigned short c, unsigned short d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned short, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned short, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned short, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<signed int, A>& cpu_trilinear(matrix<signed int, A> &e, signed int a, signed int b, signed int c, signed int d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed int, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed int, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed int, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<unsigned int, A>& cpu_trilinear(matrix<unsigned int, A> &e, unsigned int a, unsigned int b, unsigned int c, unsigned int d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned int, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned int, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned int, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<signed long long, A>& cpu_trilinear(matrix<signed long long, A> &e, signed long long a, signed long long b, signed long long c, signed long long d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed long long, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed long long, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed long long, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<unsigned long long, A>& cpu_trilinear(matrix<unsigned long long, A> &e, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned long long, cpu_avx2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned long long, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned long long, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<float, A>& cpu_trilinear(matrix<float, A> &e, float a, float b, float c, float d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_trilinear<float, cpu_avx>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse())
			kernel_fill_trilinear<float, cpu_sse>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<float, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	matrix<double, A>& cpu_trilinear(matrix<double, A> &e, double a, double b, double c, double d)
	{
		if (e.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_trilinear<double, cpu_avx>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<double, cpu_sse2>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<double, cpu_none>(e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	// Trilinear fill a tensor

	template <class A>
	tensor<signed char, A>& cpu_trilinear(tensor<signed char, A> &e, signed char a, signed char b, signed char c, signed char d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed char, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed char, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed char, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<unsigned char, A>& cpu_trilinear(tensor<unsigned char, A> &e, unsigned char a, unsigned char b, unsigned char c, unsigned char d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned char, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned char, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned char, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<signed short, A>& cpu_trilinear(tensor<signed short, A> &e, signed short a, signed short b, signed short c, signed short d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed short, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed short, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed short, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<unsigned short, A>& cpu_trilinear(tensor<unsigned short, A> &e, unsigned short a, unsigned short b, unsigned short c, unsigned short d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned short, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned short, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned short, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<signed int, A>& cpu_trilinear(tensor<signed int, A> &e, signed int a, signed int b, signed int c, signed int d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed int, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed int, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed int, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<unsigned int, A>& cpu_trilinear(tensor<unsigned int, A> &e, unsigned int a, unsigned int b, unsigned int c, unsigned int d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned int, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned int, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned int, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<signed long long, A>& cpu_trilinear(tensor<signed long long, A> &e, signed long long a, signed long long b, signed long long c, signed long long d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<signed long long, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<signed long long, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<signed long long, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<unsigned long long, A>& cpu_trilinear(tensor<unsigned long long, A> &e, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_trilinear<unsigned long long, cpu_avx2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<unsigned long long, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<unsigned long long, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<float, A>& cpu_trilinear(tensor<float, A> &e, float a, float b, float c, float d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_trilinear<float, cpu_avx>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse())
			kernel_fill_trilinear<float, cpu_sse>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<float, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

	template <class A>
	tensor<double, A>& cpu_trilinear(tensor<double, A> &e, double a, double b, double c, double d)
	{
		if (e.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_trilinear<double, cpu_avx>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_trilinear<double, cpu_sse2>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		else
			kernel_fill_trilinear<double, cpu_none>(e.batch() * e.rows(), e.columns(), e.dimension(), a, b, c, d, e.data(), e.row_size());
		return e;
	}

} // namespace core

#endif
