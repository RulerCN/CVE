/*====================================================================
Copyright (C) 2016-2016 Ruler. All rights reserved.
Author:  Ruler
Address: Nan'an District,Chongqing,China
Contact: 26105499@qq.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. The name of the author may not be used to endorse or promote
products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY RULER ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
====================================================================*/
#pragma once

#ifndef __CORE_CPU_KERNEL_SIGMOID_H__
#define __CORE_CPU_KERNEL_SIGMOID_H__

#include "../cpu_inst.h"


namespace core
{
	static const __m128  xmm_one      = _mm_set1_ps( 1.00000000F);
	static const __m128  xmm_log2e    = _mm_set1_ps( 1.44269502F);
	static const __m128i xmm_0x7f     = _mm_set1_epi32(0x0000007F);

	static const __m128  xmm_exp_min  = _mm_set1_ps(-87.3365479F);    //-126.000000/log2e;
	static const __m128  xmm_exp_max  = _mm_set1_ps( 88.3762589F);    // 127.499992/log2e;
	static const __m128  xmm_ln2_hi   = _mm_set1_ps( 0.693359375F);
	static const __m128  xmm_ln2_lo   = _mm_set1_ps(-2.12194442e-4F);
	static const __m128  xmm_exp_p1   = _mm_set1_ps( 1.000000000F);
	static const __m128  xmm_exp_p2   = _mm_set1_ps( 5.000000000e-1F);
	static const __m128  xmm_exp_p3   = _mm_set1_ps( 1.666666667e-1F);
	static const __m128  xmm_exp_p4   = _mm_set1_ps( 4.166666667e-2F);
	static const __m128  xmm_exp_p5   = _mm_set1_ps( 8.333333333e-3F);
	static const __m128  xmm_exp_p6   = _mm_set1_ps( 1.388888889e-3F);
	static const __m128  xmm_exp_p7   = _mm_set1_ps( 1.984126984e-4F);

} // namespace core

#endif
