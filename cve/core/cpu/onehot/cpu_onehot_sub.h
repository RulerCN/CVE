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

#ifndef __CORE_CPU_ONEHOT_SUB_H__
#define __CORE_CPU_ONEHOT_SUB_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../kernel/onehot/kernel_onehot_sub.h"

namespace core
{
	// Mapping an array to a matrix
	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_onehot_sub(matrix<T1, A1> &dst, matrix<T1, A1> &src, const vector<T2, A2> &onehot)
	{
		if (dst.empty() || src.empty())
			throw ::std::domain_error(matrix_not_initialized);
		if (onehot.empty())
			throw ::std::domain_error(vector_not_initialized);
		if (dst.size() != src.size())
			throw ::std::invalid_argument(invalid_size);
		if (dst.rows() != onehot.size())
			throw ::std::invalid_argument(invalid_shape);

		kernel_onehot_sub<T1, T2>()(dst.rows(), src.data(), src.row_size(), onehot.data(), dst.data());
		return dst;
	}

} // namespace core

#endif
