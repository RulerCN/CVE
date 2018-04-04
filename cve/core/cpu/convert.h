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

#ifndef __CORE_CPU_CONVERT_H__
#define __CORE_CPU_CONVERT_H__

#include "convert_int8.h"
#include "convert_uint8.h"
#include "convert_int16.h"
#include "convert_uint16.h"
#include "convert_int32.h"
#include "convert_uint32.h"
#include "convert_float.h"
#include "convert_double.h"

namespace core
{
	// Scalar data-type conversion

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<signed char, A2> &a)
	{
		return convert_int8(b, a);
	}

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<unsigned char, A2> &a)
	{
		return convert_uint8(b, a);
	}

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<signed short, A2> &a)
	{
		return convert_int16(b, a);
	}

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<unsigned short, A2> &a)
	{
		return convert_uint16(b, a);
	}

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<signed int, A2> &a)
	{
		return convert_int32(b, a);
	}

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<unsigned int, A2> &a)
	{
		return convert_uint32(b, a);
	}

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<float, A2> &a)
	{
		return convert_float(b, a);
	}

	template <class T, class A1, class A2>
	scalar<T, A1>& convert(scalar<T, A1> &b, const scalar<double, A2> &a)
	{
		return convert_double(b, a);
	}

	// Vector data-type conversion

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<signed char, A2> &a)
	{
		return convert_int8(b, a);
	}

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<unsigned char, A2> &a)
	{
		return convert_uint8(b, a);
	}

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<signed short, A2> &a)
	{
		return convert_int16(b, a);
	}

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<unsigned short, A2> &a)
	{
		return convert_uint16(b, a);
	}

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<signed int, A2> &a)
	{
		return convert_int32(b, a);
	}

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<unsigned int, A2> &a)
	{
		return convert_uint32(b, a);
	}

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<float, A2> &a)
	{
		return convert_float(b, a);
	}

	template <class T, class A1, class A2>
	vector<T, A1>& convert(vector<T, A1> &b, const vector<double, A2> &a)
	{
		return convert_double(b, a);
	}

	// Matrix data-type conversion

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<signed char, A2> &a)
	{
		return convert_int8(b, a);
	}

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<unsigned char, A2> &a)
	{
		return convert_uint8(b, a);
	}

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<signed short, A2> &a)
	{
		return convert_int16(b, a);
	}

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<unsigned short, A2> &a)
	{
		return convert_uint16(b, a);
	}

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<signed int, A2> &a)
	{
		return convert_int32(b, a);
	}

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<unsigned int, A2> &a)
	{
		return convert_uint32(b, a);
	}

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<float, A2> &a)
	{
		return convert_float(b, a);
	}

	template <class T, class A1, class A2>
	matrix<T, A1>& convert(matrix<T, A1> &b, const matrix<double, A2> &a)
	{
		return convert_double(b, a);
	}

	// Tensor data-type conversion

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<signed char, A2> &a)
	{
		return convert_int8(b, a);
	}

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<unsigned char, A2> &a)
	{
		return convert_uint8(b, a);
	}

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<signed short, A2> &a)
	{
		return convert_int16(b, a);
	}

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<unsigned short, A2> &a)
	{
		return convert_uint16(b, a);
	}

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<signed int, A2> &a)
	{
		return convert_int32(b, a);
	}

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<unsigned int, A2> &a)
	{
		return convert_uint32(b, a);
	}

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<float, A2> &a)
	{
		return convert_float(b, a);
	}

	template <class T, class A1, class A2>
	tensor<T, A1>& convert(tensor<T, A1> &b, const tensor<double, A2> &a)
	{
		return convert_double(b, a);
	}

} // namespace core

#endif
