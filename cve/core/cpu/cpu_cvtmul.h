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

#ifndef __CORE_CPU_CVTMUL_H__
#define __CORE_CPU_CVTMUL_H__

#include "convert/cpu_cvtmul_int32.h"
#include "convert/cpu_cvtmul_uint32.h"
#include "convert/cpu_cvtmul_float.h"
#include "convert/cpu_cvtmul_double.h"

namespace core
{
	// Scalar data-type conversion and scaling

	template <class T, class A1, class A2>
	scalar<signed int, A1>& cpu_cvtmul(scalar<signed int, A1> &c, const float a, const scalar<T, A2> &b)
	{
		return cpu_cvtmul_int32(c, a, b);
	}

	template <class T, class A1, class A2>
	scalar<unsigned int, A1>& cpu_cvtmul(scalar<unsigned int, A1> &c, const float a, const scalar<T, A2> &b)
	{
		return cpu_cvtmul_uint32(c, a, b);
	}

	template <class T, class A1, class A2>
	scalar<float, A1>& cpu_cvtmul(scalar<float, A1> &c, const float a, const scalar<T, A2> &b)
	{
		return cpu_cvtmul_float(c, a, b);
	}

	template <class T, class A1, class A2>
	scalar<double, A1>& cpu_cvtmul(scalar<double, A1> &c, const double a, const scalar<T, A2> &b)
	{
		return cpu_cvtmul_double(c, a, b);
	}

	// Vector data-type conversion and scaling

	template <class T, class A1, class A2>
	vector<signed int, A1>& cpu_cvtmul(vector<signed int, A1> &c, const float a, const vector<T, A2> &b)
	{
		return cpu_cvtmul_int32(c, a, b);
	}

	template <class T, class A1, class A2>
	vector<unsigned int, A1>& cpu_cvtmul(vector<unsigned int, A1> &c, const float a, const vector<T, A2> &b)
	{
		return cpu_cvtmul_uint32(c, a, b);
	}

	template <class T, class A1, class A2>
	vector<float, A1>& cpu_cvtmul(vector<float, A1> &c, const float a, const vector<T, A2> &b)
	{
		return cpu_cvtmul_float(c, a, b);
	}

	template <class T, class A1, class A2>
	vector<double, A1>& cpu_cvtmul(vector<double, A1> &c, const double a, const vector<T, A2> &b)
	{
		return cpu_cvtmul_double(c, a, b);
	}

	// Matrix data-type conversion and scaling

	template <class T, class A1, class A2>
	matrix<signed int, A1>& cpu_cvtmul(matrix<signed int, A1> &c, const float a, const matrix<T, A2> &b)
	{
		return cpu_cvtmul_int32(c, a, b);
	}

	template <class T, class A1, class A2>
	matrix<unsigned int, A1>& cpu_cvtmul(matrix<unsigned int, A1> &c, const float a, const matrix<T, A2> &b)
	{
		return cpu_cvtmul_uint32(c, a, b);
	}

	template <class T, class A1, class A2>
	matrix<float, A1>& cpu_cvtmul(matrix<float, A1> &c, const float a, const matrix<T, A2> &b)
	{
		return cpu_cvtmul_float(c, a, b);
	}

	template <class T, class A1, class A2>
	matrix<double, A1>& cpu_cvtmul(matrix<double, A1> &c, const double a, const matrix<T, A2> &b)
	{
		return cpu_cvtmul_double(c, a, b);
	}

	// Tensor data-type conversion and scaling

	template <class T, class A1, class A2>
	tensor<signed int, A1>& cpu_cvtmul(tensor<signed int, A1> &c, const float a, const tensor<T, A2> &b)
	{
		return cpu_cvtmul_int32(c, a, b);
	}

	template <class T, class A1, class A2>
	tensor<unsigned int, A1>& cpu_cvtmul(tensor<unsigned int, A1> &c, const float a, const tensor<T, A2> &b)
	{
		return cpu_cvtmul_uint32(c, a, b);
	}

	template <class T, class A1, class A2>
	tensor<float, A1>& cpu_cvtmul(tensor<float, A1> &c, const float a, const tensor<T, A2> &b)
	{
		return cpu_cvtmul_float(c, a, b);
	}

	template <class T, class A1, class A2>
	tensor<double, A1>& cpu_cvtmul(tensor<double, A1> &c, const double a, const tensor<T, A2> &b)
	{
		return cpu_cvtmul_double(c, a, b);
	}

} // namespace core

#endif
