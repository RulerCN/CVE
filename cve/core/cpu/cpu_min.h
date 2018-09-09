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

#ifndef __CORE_CPU_MIN_H__
#define __CORE_CPU_MIN_H__

#include "min/cpu_min_x.h"
#include "min/cpu_min_y.h"
#include "min/cpu_min_z.h"
#include "min/cpu_min_xy.h"
#include "min/cpu_min_yz.h"
#include "min/cpu_min_xyz.h"

namespace core
{
	// Computes the min of elements across dimensions of a vector

	template <class T1, class T2, class A1, class A2>
	T1& cpu_min(T1 &b, const vector<T2, A2> &a, axis_type axis = axis_x)
	{
		switch (axis)
		{
		case axis_x:
			cpu_min_x(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_min(vector<T1, A1> &b, const vector<T2, A2> &a, axis_type axis = axis_x)
	{
		switch (axis)
		{
		case axis_x:
			cpu_min_x(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	// Computes the min of elements across dimensions of a matrix

	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_min(vector<T1, A1> &b, const matrix<T2, A2> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_min_x(b, a);
			break;
		case axis_y:
			cpu_min_y(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	T1& cpu_min(T1 &b, const matrix<T2, A2> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_xy:
			cpu_min_xy(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_min(matrix<T1, A1> &b, const matrix<T2, A2> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_min_x(b, a);
			break;
		case axis_y:
			cpu_min_y(b, a);
			break;
		case axis_xy:
			cpu_min_xy(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	// Computes the min of elements across dimensions of a tensor

	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_min(matrix<T1, A1> &b, const tensor<T2, A2> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_min_x(b, a);
			break;
		case axis_y:
			cpu_min_y(b, a);
			break;
		case axis_z:
			cpu_min_z(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_min(vector<T1, A1> &b, const tensor<T2, A2> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_xy:
			cpu_min_xy(b, a);
			break;
		case axis_yz:
			cpu_min_yz(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	T1& cpu_min(T1 &b, const tensor<T2, A2> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_xyz:
			cpu_min_xyz(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	tensor<T1, A1>& cpu_min(tensor<T1, A1> &b, const tensor<T2, A2> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_min_x(b, a);
			break;
		case axis_y:
			cpu_min_y(b, a);
			break;
		case axis_z:
			cpu_min_z(b, a);
			break;
		case axis_xy:
			cpu_min_xy(b, a);
			break;
		case axis_yz:
			cpu_min_yz(b, a);
			break;
		case axis_xyz:
			cpu_min_xyz(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

} // namespace core

#endif
