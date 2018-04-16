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

#ifndef __CORE_CPU_CONVERT_SCALE_H__
#define __CORE_CPU_CONVERT_SCALE_H__

#include "cpu_convert_scale_float.h"
#include "cpu_convert_scale_double.h"

namespace core
{
	// Scalar data-type conversion and scaling

	template <class T, class A1, class A2>
	scalar<float, A1>& cpu_convert_scale(scalar<float, A1> &b, const scalar<T, A2> &a, float scale)
	{
		return cpu_convert_scale_float(b, a, scale);
	}

	template <class T, class A1, class A2>
	scalar<double, A1>& cpu_convert_scale(scalar<double, A1> &b, const scalar<T, A2> &a, double scale)
	{
		return cpu_convert_scale_double(b, a, scale);
	}

	// Vector data-type conversion and scaling

	template <class T, class A1, class A2>
	vector<float, A1>& cpu_convert_scale(vector<float, A1> &b, const vector<T, A2> &a, float scale)
	{
		return cpu_convert_scale_float(b, a, scale);
	}

	template <class T, class A1, class A2>
	vector<double, A1>& cpu_convert_scale(vector<double, A1> &b, const vector<T, A2> &a, double scale)
	{
		return cpu_convert_scale_double(b, a, scale);
	}

	// Matrix data-type conversion and scaling

	template <class T, class A1, class A2>
	matrix<float, A1>& cpu_convert_scale(matrix<float, A1> &b, const matrix<T, A2> &a, float scale)
	{
		return cpu_convert_scale_float(b, a, scale);
	}

	template <class T, class A1, class A2>
	matrix<double, A1>& cpu_convert_scale(matrix<double, A1> &b, const matrix<T, A2> &a, double scale)
	{
		return cpu_convert_scale_double(b, a, scale);
	}

	// Tensor data-type conversion and scaling

	template <class T, class A1, class A2>
	tensor<float, A1>& cpu_convert_scale(tensor<float, A1> &b, const tensor<T, A2> &a, float scale)
	{
		return cpu_convert_scale_float(b, a, scale);
	}

	template <class T, class A1, class A2>
	tensor<double, A1>& cpu_convert_scale(tensor<double, A1> &b, const tensor<T, A2> &a, double scale)
	{
		return cpu_convert_scale_double(b, a, scale);
	}

} // namespace core

#endif
