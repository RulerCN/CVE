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

#ifndef __CORE_CPU_MAX_H__
#define __CORE_CPU_MAX_H__

#include "max/cpu_max_x.h"
#include "max/cpu_max_y.h"
#include "max/cpu_max_z.h"
#include "max/cpu_max_xy.h"
#include "max/cpu_max_yz.h"
#include "max/cpu_max_xyz.h"

namespace core
{
	// Computes the max of elements across dimensions of a vector

	template <class T, class A>
	T& cpu_max(T &b, const vector<T, A> &a, axis_type axis = axis_x)
	{
		switch (axis)
		{
		case axis_x:
			cpu_max_x(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T, class A>
	vector<T, A>& cpu_max(vector<T, A> &b, const vector<T, A> &a, axis_type axis = axis_x)
	{
		switch (axis)
		{
		case axis_x:
			cpu_max_x(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	// Computes the max of elements across dimensions of a matrix

	template <class T, class A>
	vector<T, A>& cpu_max(vector<T, A> &b, const matrix<T, A> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_max_x(b, a);
			break;
		case axis_y:
			cpu_max_y(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T, class A>
	T& cpu_max(T &b, const matrix<T, A> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_xy:
			cpu_max_xy(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T, class A>
	matrix<T, A>& cpu_max(matrix<T, A> &b, const matrix<T, A> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_max_x(b, a);
			break;
		case axis_y:
			cpu_max_y(b, a);
			break;
		case axis_xy:
			cpu_max_xy(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	// Computes the max of elements across dimensions of a tensor

	template <class T, class A>
	matrix<T, A>& cpu_max(matrix<T, A> &b, const tensor<T, A> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_max_x(b, a);
			break;
		case axis_y:
			cpu_max_y(b, a);
			break;
		case axis_z:
			cpu_max_z(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T, class A>
	vector<T, A>& cpu_max(vector<T, A> &b, const tensor<T, A> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_xy:
			cpu_max_xy(b, a);
			break;
		case axis_yz:
			cpu_max_yz(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T, class A>
	T& cpu_max(T &b, const tensor<T, A> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_xyz:
			cpu_max_xyz(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

	template <class T, class A>
	tensor<T, A>& cpu_max(tensor<T, A> &b, const tensor<T, A> &a, axis_type axis)
	{
		switch (axis)
		{
		case axis_x:
			cpu_max_x(b, a);
			break;
		case axis_y:
			cpu_max_y(b, a);
			break;
		case axis_z:
			cpu_max_z(b, a);
			break;
		case axis_xy:
			cpu_max_xy(b, a);
			break;
		case axis_yz:
			cpu_max_yz(b, a);
			break;
		case axis_xyz:
			cpu_max_xyz(b, a);
			break;
		default:
			throw ::std::invalid_argument(invalid_axis_parameters);
		}
		return b;
	}

} // namespace core

#endif
