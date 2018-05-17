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

#ifndef __CORE_CPU_KERNEL_MAPPING_H__
#define __CORE_CPU_KERNEL_MAPPING_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template kernel_mapping
	template<class T1, class T2>
	struct kernel_mapping
	{
		// C[i] = A[B[i]]
		void operator()(size_t n, const T1 *a, const T2 *b, T1 *c) const
		{
			constexpr size_t block_n = 8;
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_n = n - aligned_n;
			for (size_t i = 0; i < aligned_n; i += block_n)
			{
				c[0] = a[b[0]];
				c[1] = a[b[1]];
				c[2] = a[b[2]];
				c[3] = a[b[3]];
				c[4] = a[b[4]];
				c[5] = a[b[5]];
				c[6] = a[b[6]];
				c[7] = a[b[7]];
				b += block_n;
				c += block_n;
			}
			for (size_t i = 0; i < surplus_n; ++i)
				c[i] = a[b[i]];
		}
	};

} // namespace core

#endif
