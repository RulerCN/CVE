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
	// Subtraction matrix and one-hot vector

	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_onehot_sub(matrix<T1, A1> &dst, const vector<T2, A2> &onehot)
	{
		if (dst.empty())
			throw ::std::domain_error(matrix_not_initialized);
		if (onehot.empty())
			throw ::std::domain_error(vector_not_initialized);
		if (dst.rows() != onehot.size())
			throw ::std::invalid_argument(invalid_shape);

		kernel_onehot_sub(dst.rows(), dst.row_size(), dst.data(), onehot.data());
		return dst;
	}

	template <class T1, class T2, class A1, class A2>
	matrix<T1, A1>& cpu_onehot_sub(matrix<T1, A1> &dst, const matrix<T1, A1> &src, const vector<T2, A2> &onehot)
	{
		if (onehot.empty())
			throw ::std::domain_error(vector_not_initialized);
		if (dst.rows() != onehot.size())
			throw ::std::invalid_argument(invalid_shape);

		dst.fill(src);
		kernel_onehot_sub(dst.rows(), dst.row_size(), dst.data(), onehot.data());
		return dst;
	}

	// Subtraction tensor and one-hot matrix

	template <class T1, class T2, class A1, class A2>
	tensor<T1, A1>& cpu_onehot_sub(tensor<T1, A1> &dst, const matrix<T2, A2> &onehot)
	{
		if (dst.empty())
			throw ::std::domain_error(tensor_not_initialized);
		if (onehot.empty())
			throw ::std::domain_error(matrix_not_initialized);
		if (dst.batch() != onehot.rows())
			throw ::std::invalid_argument(invalid_shape);

		kernel_onehot_sub(dst.batch(), dst.rows(), dst.row_size(), dst.data(), onehot.data());
		return dst;
	}

	template <class T1, class T2, class A1, class A2>
	tensor<T1, A1>& cpu_onehot_sub(tensor<T1, A1> &dst, const tensor<T1, A1> &src, const matrix<T2, A2> &onehot)
	{
		if (onehot.empty())
			throw ::std::domain_error(matrix_not_initialized);
		if (dst.batch() != onehot.rows())
			throw ::std::invalid_argument(invalid_shape);

		dst.fill(src);
		kernel_onehot_sub(dst.batch(), dst.rows(), dst.row_size(), dst.data(), onehot.data());
		return dst;
	}

	// Subtraction tensor and one-hot tensor

	template <class T1, class T2, class A1, class A2>
	tensor<T1, A1>& cpu_onehot_sub(tensor<T1, A1> &dst, const tensor<T2, A2> &onehot)
	{
		if (dst.empty() || onehot.empty())
			throw ::std::domain_error(tensor_not_initialized);
		if (dst.batch() != onehot.batch() || dst.rows() != onehot.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		kernel_onehot_sub(dst.batch(), dst.rows(), dst.row_size(), dst.data(), onehot.data());
		return dst;
	}

	template <class T1, class T2, class A1, class A2>
	tensor<T1, A1>& cpu_onehot_sub(tensor<T1, A1> &dst, const tensor<T1, A1> &src, const tensor<T2, A2> &onehot)
	{
		if (onehot.empty())
			throw ::std::domain_error(tensor_not_initialized);
		if (dst.batch() != onehot.batch() || dst.rows() != onehot.matrix_size())
			throw ::std::invalid_argument(invalid_shape);

		dst.fill(src);
		kernel_onehot_sub(dst.batch(), dst.rows(), dst.row_size(), dst.data(), onehot.data());
		return dst;
	}

} // namespace core

#endif
