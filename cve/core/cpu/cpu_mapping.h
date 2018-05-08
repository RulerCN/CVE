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

#ifndef __CORE_CPU_MAPPING_H__
#define __CORE_CPU_MAPPING_H__

#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include "kernel/kernel_mapping.h"

namespace core
{
	// Mapping an array to a scalar
	template <class T1, class T2, class A1, class A2>
	scalar<T1, A1>& cpu_mapping(scalar<T1, A1> &dst, const T1 *p, const scalar<T2, A2> &idx)
	{
		if (dst.empty() || idx.empty())
			throw ::std::domain_error(scalar_not_initialized);
		if (p == nullptr)
			throw ::std::invalid_argument(invalid_pointer);
		if (dst.size() != idx.size())
			throw ::std::invalid_argument(scalar_different_size);

		kernel_mapping<T1, T2>()(dst.size(), p, idx.data(), dst.data());
		return dst;
	}

	// Mapping an array to a vector
	template <class T1, class T2, class A1, class A2>
	vector<T1, A1>& cpu_mapping(vector<T1, A1> &dst, const T1 *p, const vector<T2, A2> &idx)
	{
		if (dst.empty() || idx.empty())
			throw ::std::domain_error(vector_not_initialized);
		if (p == nullptr)
			throw ::std::invalid_argument(invalid_pointer);
		if (dst.size() != idx.size())
			throw ::std::invalid_argument(vector_different_size);

		kernel_mapping<T1, T2>()(dst.size(), p, idx.data(), dst.data());
		return dst;
	}

	// Mapping an array to a matrix
	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_mapping(matrix<T1, A1> &dst, const T1 *p, const matrix<T2, A2> &idx)
	{
		if (dst.empty() || idx.empty())
			throw ::std::domain_error(matrix_not_initialized);
		if (p == nullptr)
			throw ::std::invalid_argument(invalid_pointer);
		if (dst.size() != idx.size())
			throw ::std::invalid_argument(matrix_different_size);

		kernel_mapping<T1, T2>()(dst.size(), p, idx.data(), dst.data());
		return dst;
	}

	// Mapping an array to a tensor
	template <class T1, class T2, class A1, class A2>
	tensor<T1, A1>& cpu_mapping(tensor<T1, A1> &dst, const T1 *p, const tensor<T2, A2> &idx)
	{
		if (dst.empty() || idx.empty())
			throw ::std::domain_error(tensor_not_initialized);
		if (p == nullptr)
			throw ::std::invalid_argument(invalid_pointer);
		if (dst.size() != idx.size())
			throw ::std::invalid_argument(tensor_different_size);

		kernel_mapping<T1, T2>()(dst.size(), p, idx.data(), dst.data());
		return dst;
	}

} // namespace core

#endif
