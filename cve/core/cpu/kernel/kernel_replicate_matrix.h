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

#ifndef __CORE_CPU_KERNEL_REPLICATE_MATRIX_H__
#define __CORE_CPU_KERNEL_REPLICATE_MATRIX_H__

#include <cstring>

namespace core
{
	// Function template kernel_replicate_matrix
	template<class T>
	void kernel_replicate_matrix(size_t m, size_t n, const T *a, size_t rsa, size_t rows, T *b, size_t rsb)
	{
		size_t size = rsa * sizeof(T);
		const T *ptr_a = a;
		T *ptr, *ptr_b = b;

		for (size_t j = 0; j < rows; ++j)
		{
			ptr = ptr_b;
			for (size_t i = 0; i < n; ++i)
			{
				::std::memcpy(ptr, ptr_a, size);
				ptr += rsa;
			}
			ptr_a += rsa;
			ptr_b += rsb;
		}
		size = n * size;
		for (size_t k = 1; k < m; ++k)
		{
			ptr = b;
			for (size_t j = 0; j < rows; ++j)
			{
				::std::memcpy(ptr_b, ptr, size);
				ptr += rsb;
				ptr_b += rsb;
			}
		}
	}

} // namespace core

#endif
