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

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/min/kernel_min.h"

namespace core
{
	// Computes the min of elements of a vector

	template <class A>
	signed char& cpu_min_x(signed char &b, const vector<signed char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = int8_max;
		if (cpu_inst::is_support_avx2())
			kernel_min<signed char, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_min<signed char, 8, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_min<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

} // namespace core

#endif
