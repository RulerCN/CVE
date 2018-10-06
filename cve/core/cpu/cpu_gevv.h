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

#ifndef __CORE_CPU_GEVV_H__
#define __CORE_CPU_GEVV_H__

#include "gevv/cpu_addvv.h"

namespace core
{
	// The multiplication of the column vector and the row vector

	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_gevv(matrix<T, A> &c, const vector<T, A1> &a, const vector<T, A2> &b)
	{
		c.fill(T(0));
		return cpu_addvv(c, a, b);
	}

	template <class T, class A, class A1, class A2>
	matrix<T, A>& cpu_gevv(matrix<T, A> &d, const vector<T, A1> &a, const vector<T, A2> &b, const vector<T, A> &c)
	{
		d.fill(c);
		return cpu_addvv(d, a, b);
	}

	// The multiplication of the matrix and the matrix

	template <class T, class A, class A1, class A2>
	tensor<T, A>& cpu_gevm(tensor<T, A> &c, const matrix<T, A1> &a, const matrix<T, A2> &b)
	{
		c.fill(T(0));
		return cpu_addvv(c, a, b);
	}

	template <class T, class A, class A1, class A2>
	tensor<T, A>& cpu_gevm(tensor<T, A> &d, const matrix<T, A1> &a, const matrix<T, A2> &b, const tensor<T, A> &c)
	{
		d.fill(c);
		return cpu_addvv(d, a, b);
	}

	// The multiplication of the tensor and the tensor

	template <class T, class A, class A1, class A2>
	tensor<T, A>& cpu_gevm(tensor<T, A> &c, const tensor<T, A1> &a, const tensor<T, A2> &b)
	{
		c.fill(T(0));
		return cpu_addvv(c, a, b);
	}

	template <class T, class A, class A1, class A2>
	tensor<T, A>& cpu_gevm(tensor<T, A> &d, const tensor<T, A1> &a, const tensor<T, A2> &b, const tensor<T, A1> &c)
	{
		d.fill(c);
		return cpu_addvv(d, a, b);
	}

} // namespace core

#endif
