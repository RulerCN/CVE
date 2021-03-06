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

#ifndef __CORE_H__
#define __CORE_H__

#include "sample_allocator.h"
#include "allocator.h"
#include "sample_allocator.h"
#include "scalar.h"
#include "vector.h"
#include "matrix.h"
#include "tensor.h"
#include "rb_tree.h"
//#include "sb_tree.h"
#include "tree.h"

#include "cpu/cpu_math.h"
#include "cpu/cpu_fill.h"
#include "cpu/cpu_repeat.h"
#include "cpu/cpu_transpose.h"
#include "cpu/cpu_convert.h"
#include "cpu/cpu_border.h"
#include "cpu/cpu_sliding.h"
#include "cpu/cpu_mapping.h"
#include "cpu/cpu_arithmetic.h"
#include "cpu/cpu_onehot.h"
#include "cpu/cpu_logic.h"
#include "cpu/cpu_max.h"
#include "cpu/cpu_mean.h"
#include "cpu/cpu_min.h"
#include "cpu/cpu_sum.h"
#include "cpu/cpu_gemm.h"
#include "cpu/cpu_gevm.h"
#include "cpu/cpu_gevv.h"
#include "cpu/cpu_gtvv.h"

#include "cpu/cpu_get_element.h"

#endif
