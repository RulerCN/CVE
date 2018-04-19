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

#ifndef __CORE_CPU_BORDER_H__
#define __CORE_CPU_BORDER_H__

#include "cpu_border_replicte.h"

namespace core
{
	//template <class A>
	//matrix<size_t, A>& cpu_border_index_x(matrix<size_t, A> &index, size_t left, size_t right)
	//{
	//	if (index.empty())
	//		throw ::std::invalid_argument(matrix_not_initialized);
	//	if (left + right >= index.columns())
	//		throw ::std::invalid_argument(invalid_border_size);

	//	//size_t rows, size_t columns, size_t dimension,

	//	for (size_t i = 0; i < count; ++i)
	//	{
	//		buffer[i] = value;
	//		value += delta;
	//	}

	//	return index;
	//}

	////static constexpr border_type        border_constant    = 0x00;                       /* iiii|abcdefgh|iiii */
	////static constexpr border_type        border_replicte    = 0x01;                       /* aaaa|abcdefgh|hhhh */
	////static constexpr border_type        border_reflect     = 0x02;                       /* dcba|abcdefgh|hgfe */
	////static constexpr border_type        border_reflect101  = 0x03;                       /* edcb|abcdefgh|gfed */
	////static constexpr border_type        border_wrap        = 0x04;                       /* efgh|abcdefgh|abcd */

	//template <class A>
	//matrix<size_t, A>& cpu_border_replicte_x(matrix<size_t, A> &index, size_t border_left, size_t border_top, size_t border_right, size_t border_bottom)
	//{
	//	if (index.empty())
	//		throw ::std::invalid_argument(matrix_not_initialized);
	//	if (border_left + border_right >= index.columns())
	//		throw ::std::invalid_argument(invalid_border_size);

	//	size_t rows = index.rows() - border_top - border_bottom;
	//	size_t cols = index.columns() - border_left - border_right;
	//	size_t dim = index.dimension();
	//	size_t row_size = cols * dim;
	//	size_t loop = left / cols;
	//	size_t remain = (left % cols) * dim;

	//	if (loop & 1)
	//	{
	//		value = row_size - remain;
	//		for (size_t i = 0; i < remain; i += dim)
	//			ptr[i] = value + i;
	//		ptr += remain - dim;
	//		value = row_size - dim;
	//		for (size_t i = 0; i < row_size; i += dim)
	//			ptr[i] = value - i;
	//		ptr += row_size - dim;
	//		--loop;
	//	}
	//	else
	//	{
	//		value = remain - dim;
	//		for (size_t i = 0; i < remain; i += dim)
	//			ptr[i] = value - i;
	//		ptr += remain - dim;
	//	}
	//	for (size_t i = 0; i < loop; i += 2)
	//	{
	//		for (size_t i = 0; i < row_size; i += dim)
	//			ptr[i] = i;
	//		ptr += row_size;
	//		value = row_size - dim;
	//		for (size_t i = 0; i < row_size; i += dim)
	//			ptr[i] = value - i;
	//		ptr += row_size - dim;
	//	}
	//}

} // namespace core

#endif
