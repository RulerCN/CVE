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

#ifndef __CORE_ERROR_H__
#define __CORE_ERROR_H__

#include <stdexcept>

namespace core
{
	//----------------------------------------------------------------
	// +exception
	// |-----bad_alloc:              Exception thrown on failure allocating memory
	// |-----bad_cast:               Exception thrown on failure to dynamic cast
	// |-----bad_exception:          Exception thrown by unexpected handler
	// |-----bad_function_call:      Exception thrown on bad call
	// |-----bad_typeid:             Exception thrown on typeid of null pointer
	// |-----bad_weak_ptr:           Bad weak pointer
	// |-----ios_base::failure:      Base class for stream exceptions
	// |----+logic_error:            Logic error exception
	// |    |----domain_error
	// |    |----invalid_argument
	// |    |----length_error
	// |    |----out_of_range
	// |    |----future_error
	// |
	// |----+runtime_error:          Runtime error exception
	//      |--range_error
	//      |--overflow_error
	//      |--underflow_error
	//      |--system_error

	// bad_alloc
	static const char not_enough_memory[]         = "Not enough storage.";
	// domain_error
	static const char scalar_is_initialized[]     = "Scalar is initialized.";
	static const char scalar_not_initialized[]    = "Scalar not initialized.";
	static const char vector_is_initialized[]     = "Vector is initialized.";
	static const char vector_not_initialized[]    = "Vector not initialized.";
	static const char matrix_is_initialized[]     = "Matrix is initialized.";
	static const char matrix_not_initialized[]    = "Matrix not initialized.";
	static const char tensor_is_initialized[]     = "Tensor is initialized.";
	static const char tensor_not_initialized[]    = "Tensor not initialized.";
	static const char cache_is_initialized[]      = "Cache is initialized.";
	static const char cache_not_initialized[]     = "Cache not initialized.";
	static const char scalar_different_size[]     = "Unable to operate scalars of different size.";
	static const char vector_different_size[]     = "Unable to operate vectors of different size.";
	static const char matrix_different_size[]     = "Unable to operate matrixs of different size.";
	static const char tensor_different_size[]     = "Unable to operate tensors of different size.";
	static const char sample_unequal_number[]     = "Unequal number of images and labels.";
	// invalid_argument
	static const char invalid_power[]             = "Invalid power.";
	static const char invalid_dimension[]         = "Invalid dimension.";
	static const char invalid_pointer[]           = "Invalid pointer.";
	static const char invalid_length[]            = "Invalid length.";
	static const char invalid_size[]              = "Invalid size.";
	static const char invalid_shape[]             = "Invalid shape.";
	static const char invalid_border_size[]       = "Invalid border size.";
	static const char invalid_initializer_list[]  = "Invalid initializer list.";
	static const char invalid_iterator_distance[] = "Invalid distance between iterators.";
	static const char invalid_mode_parameters[]   = "Invalid mode parameters.";
	static const char scalar_invalid_size[]       = "Invalid size of the scalar.";
	static const char vector_invalid_size[]       = "Invalid size of the vector.";
	static const char matrix_invalid_size[]       = "Invalid size of the matrix.";
	static const char tensor_invalid_size[]       = "Invalid size of the tensor.";
	static const char cache_invalid_size[]        = "Invalid size of the cache.";
	// out_of_range
	static const char scalar_out_of_range[]       = "Scalar subscript out of range.";
	static const char vector_out_of_range[]       = "Vector subscript out of range.";
	static const char matrix_out_of_range[]       = "Matrix subscript out of range.";
	static const char tensor_out_of_range[]       = "Tensor subscript out of range.";

} // namespace core

#endif
