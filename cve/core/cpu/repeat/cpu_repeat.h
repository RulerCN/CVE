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

#ifndef __CORE_CPU_REPEAT_H__
#define __CORE_CPU_REPEAT_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/repeat/kernel_repeat.h"

namespace core
{
	// Repeat and tile a vector

	template <class T, class A>
	vector<T, A>& cpu_repeat(vector<T, A> &b, const vector<T, A> &a, size_t n)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != n * a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_repeat(n, a.data(), a.size(), b.data());
		return b;
	}

	template <class T, class A>
	matrix<T, A>& cpu_repeat(matrix<T, A> &b, const vector<T, A> &a, size_t m, size_t n)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != m * n * a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_repeat(m * n, a.data(), a.size(), b.data());
		return b;
	}

	template <class T, class A>
	tensor<T, A>& cpu_repeat(tensor<T, A> &b, const vector<T, A> &a, size_t l, size_t m, size_t n)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != l * m * n * a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_repeat(l * m * n, a.data(), a.size(), b.data());
		return b;
	}

} // namespace core

#endif
