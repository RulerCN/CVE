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

#ifndef __CORE_CPU_GTVV_H__
#define __CORE_CPU_GTVV_H__

#include "gtvv/cpu_addtvv.h"

namespace core
{
	// The multiplication of the column vector and the row vector

	template <class T, class A>
	matrix<T, A>& cpu_gtvv(matrix<T, A> &c, const vector<T, A> &a, const vector<T, A> &b)
	{
		c.fill(T(0));
		return cpu_addtvv(c, a, b);
	}

	template <class T, class A>
	matrix<T, A>& cpu_gtvv(matrix<T, A> &d, const vector<T, A> &a, const vector<T, A> &b, const matrix<T, A> &c)
	{
		d.fill(c);
		return cpu_addtvv(d, a, b);
	}

	// The multiplication of the column vector and the row vector

	template <class T, class A>
	tensor<T, A>& cpu_gtvv(tensor<T, A> &c, const matrix<T, A> &a, const matrix<T, A> &b)
	{
		c.fill(T(0));
		return cpu_addtvv(c, a, b);
	}

	template <class T, class A>
	tensor<T, A>& cpu_gtvv(tensor<T, A> &d, const matrix<T, A> &a, const matrix<T, A> &b, const tensor<T, A> &c)
	{
		d.fill(c);
		return cpu_addtvv(d, a, b);
	}

	// The multiplication of the column vector and the row vector

	template <class T, class A>
	tensor<T, A>& cpu_gtvv(tensor<T, A> &c, const tensor<T, A> &a, const tensor<T, A> &b)
	{
		c.fill(T(0));
		return cpu_addtvv(c, a, b);
	}

	template <class T, class A>
	tensor<T, A>& cpu_gtvv(tensor<T, A> &d, const tensor<T, A> &a, const tensor<T, A> &b, const tensor<T, A> &c)
	{
		d.fill(c);
		return cpu_addtvv(d, a, b);
	}

} // namespace core

#endif
