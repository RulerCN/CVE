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

#ifndef __CORE_CPU_REDUCE_MEAN_H__
#define __CORE_CPU_REDUCE_MEAN_H__

#include "sum/cpu_sum_x_int32.h"
#include "sum/cpu_sum_x_float.h"
#include "sum/cpu_sum_x_double.h"
#include "sum/cpu_sum_y_int32.h"
#include "sum/cpu_sum_y_float.h"
#include "sum/cpu_sum_y_double.h"
#include "sum/cpu_sum_z_int32.h"
#include "sum/cpu_sum_z_float.h"
#include "sum/cpu_sum_z_double.h"
#include "sum/cpu_sum_xy_int32.h"
#include "sum/cpu_sum_xy_float.h"
#include "sum/cpu_sum_xy_double.h"
#include "sum/cpu_sum_yz_int32.h"
#include "sum/cpu_sum_yz_float.h"
#include "sum/cpu_sum_yz_double.h"
#include "sum/cpu_sum_xyz_int32.h"
#include "sum/cpu_sum_xyz_float.h"
#include "sum/cpu_sum_xyz_double.h"
#include "cpu_arithmetic.h"

namespace core
{
	// Computes the sum of elements across dimensions of a vector

	template <class T1, class T2, class A1, class A2>
	T1& cpu_mean(T1 &b, const vector<T2, A2> &a, axis_type axis = axis_x)
	{
		T1 scale;
		switch (axis)
		{
		case axis_x:
			scale = T1(1) / a.size();
			cpu_sum_x(b, a);
			b *= scale;
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_mean(vector<T1, A1> &b, const vector<T2, A2> &a, axis_type axis = axis_x)
	{
		T1 scale;
		switch (axis)
		{
		case axis_x:
			scale = T1(1) / a.size();
			cpu_sum_x(b, a);
			cpu_mul(b, scale, b);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	// Computes the sum of elements across dimensions of a matrix

	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_mean(vector<T1, A1> &b, const matrix<T2, A2> &a, axis_type axis)
	{
		T1 scale;
		switch (axis)
		{
		case axis_x:
			scale = T1(1) / a.row_size();
			cpu_sum_x(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_y:
			scale = T1(1) / a.rows();
			cpu_sum_y(b, a);
			cpu_mul(b, scale, b);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	T1& cpu_mean(T1 &b, const matrix<T2, A2> &a, axis_type axis)
	{
		T1 scale;
		switch (axis)
		{
		case axis_xy:
			scale = T1(1) / a.size();
			cpu_sum_xy(b, a);
			b *= scale;
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_mean(matrix<T1, A1> &b, const matrix<T2, A2> &a, axis_type axis)
	{
		T1 scale;
		switch (axis)
		{
		case axis_x:
			scale = T1(1) / a.row_size();
			cpu_sum_x(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_y:
			scale = T1(1) / a.rows();
			cpu_sum_y(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_xy:
			scale = T1(1) / a.size();
			cpu_sum_xy(b, a);
			cpu_mul(b, scale, b);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	// Computes the sum of elements across dimensions of a tensor

	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_mean(matrix<T1, A1> &b, const tensor<T2, A2> &a, axis_type axis)
	{
		T1 scale;
		switch (axis)
		{
		case axis_x:
			scale = T1(1) / a.row_size();
			cpu_sum_x(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_y:
			scale = T1(1) / a.rows();
			cpu_sum_y(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_z:
			scale = T1(1) / a.batch();
			cpu_sum_z(b, a);
			cpu_mul(b, scale, b);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_mean(vector<T1, A1> &b, const tensor<T2, A2> &a, axis_type axis)
	{
		T1 scale;
		switch (axis)
		{
		case axis_xy:
			scale = T1(1) / a.matrix_size();
			cpu_sum_xy(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_yz:
			scale = T1(1) / (a.batch() * a.rows());
			cpu_sum_yz(b, a);
			cpu_mul(b, scale, b);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	T1& cpu_mean(T1 &b, const tensor<T2, A2> &a, axis_type axis)
	{
		T1 scale;
		switch (axis)
		{
		case axis_xyz:
			scale = T1(1) / a.size();
			cpu_sum_xyz(b, a);
			b *= scale;
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T1, class T2, class A1, class A2>
	tensor<T1, A1>& cpu_mean(tensor<T1, A1> &b, const tensor<T2, A2> &a, axis_type axis)
	{
		T1 scale;
		switch (axis)
		{
		case axis_x:
			scale = T1(1) / a.row_size();
			cpu_sum_x(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_y:
			scale = T1(1) / a.rows();
			cpu_sum_y(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_z:
			scale = T1(1) / a.batch();
			cpu_sum_z(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_xy:
			scale = T1(1) / a.matrix_size();
			cpu_sum_xy(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_yz:
			scale = T1(1) / (a.batch() * a.rows());
			cpu_sum_yz(b, a);
			cpu_mul(b, scale, b);
			break;
		case axis_xyz:
			scale = T1(1) / a.size();
			cpu_sum_xyz(b, a);
			cpu_mul(b, scale, b);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

} // namespace core

#endif
