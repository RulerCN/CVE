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

#ifndef __CORE_CPU_BORDER_INDEX_H__
#define __CORE_CPU_BORDER_INDEX_H__

#include "../matrix.h"

namespace core
{
	template <class A>
	matrix<size_t, A>& cpu_border_index(matrix<size_t, A> &index, size_t rows, size_t columns, size_t dimension, size_t border)
	{
		//if (b.empty() || a.empty())
		//	throw ::std::invalid_argument(matrix_not_initialized);
		//if (b.size() != a.size())
		//	throw ::std::invalid_argument(invalid_size);

		//if (global::is_support_avx2())
		//	kernel_convert_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data());
		//else if (global::is_support_sse2())
		//	kernel_convert_float<unsigned char, inst_sse2>()(a.size(), a.data(), b.data());
		//else
		//	kernel_convert_float<unsigned char, inst_none>()(a.size(), a.data(), b.data());
		//return b;
	}

} // namespace core

#endif
