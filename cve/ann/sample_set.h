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

#ifndef __ANN_TRAINING_SAMPLE_H__
#define __ANN_TRAINING_SAMPLE_H__

#include <iostream>
#include <fstream>
#include "sample_base.h"

namespace ann
{
	// Class template train_sample
	template <class T1, class T2, class Allocator = ::core::allocator<void> >
	class sample_set : public sample_base<Allocator>
	{
	public:
		// types:

		typedef typename Allocator::template rebind<T1>::other data_allocator_type;
		typedef typename Allocator::template rebind<T2>::other labels_allocator_type;

		typedef ::core::scalar<T1, data_allocator_type>        data_scalar_type;
		typedef ::core::vector<T1, data_allocator_type>        data_vector_type;
		typedef ::core::matrix<T1, data_allocator_type>        data_matrix_type;
		typedef ::core::tensor<T1, data_allocator_type>        data_tensor_type;
		typedef const data_scalar_type                         const_data_scalar_type;
		typedef	const data_vector_type                         const_data_vector_type;
		typedef	const data_matrix_type                         const_data_matrix_type;
		typedef	const data_tensor_type                         const_data_tensor_type;
		typedef ::core::scalar<T2, labels_allocator_type>      labels_scalar_type;
		typedef ::core::vector<T2, labels_allocator_type>      labels_vector_type;
		typedef ::core::matrix<T2, labels_allocator_type>      labels_matrix_type;
		typedef ::core::tensor<T2, labels_allocator_type>      labels_tensor_type;
		typedef const labels_scalar_type                       const_labels_scalar_type;
		typedef	const labels_vector_type                       const_labels_vector_type;
		typedef	const labels_matrix_type                       const_labels_matrix_type;
		typedef	const labels_tensor_type                       const_labels_tensor_type;

		// construct/copy/destroy:

		sample_set(void)
		{}

		sample_set(size_type batch, size_t rows, size_t columns, size_t dimension)
		{
			assign(batch, rows, columns, dimension);
		}

		void assign(size_type batch, size_t rows, size_t columns, size_t dimension)
		{
			constexpr size_t one = 1;
			data.assign(batch, rows, columns, dimension);
			labels.assign(batch, one, one, one);
		}

		// capacity:

		bool empty(void) const noexcept
		{
			return (data.empty() || labels.empty());
		}

		// Return a batch of samples

		template<class U1, class U2, class A>
		void next_batch(sample_set<U1, U2, A> &batch_samples)
		{
			if (empty() || batch_samples.empty())
				throw ::std::domain_error(::core::sample_set_not_initialized);

			size_t batch_size = batch_data.batch();
			for (size_t i = 0; i < batch_size; ++i)
			{
				size_t index = this->next();
				::core::cpu_get_element(batch_data.at(i), data, index);
				::core::cpu_get_element(batch_labels.at(i), labels, index);
			}
		}

		template<class U1, class U2, class A1, class A2>
		void next_batch(::core::tensor<U1, A1> &batch_data, ::core::tensor<U2, A2> &batch_labels, U1 scale)
		{
			if (batch_data.batch() != batch_labels.length())
				throw ::std::invalid_argument(::core::sample_unequal_number);

			size_t batch_size = batch_data.batch();
			for (size_t i = 0; i < batch_size; ++i)
			{
				size_t index = this->next();
				::core::cpu_get_element(batch_data.at(i), data, index, scale);
				::core::cpu_get_element(batch_labels.at(i), labels, index);
			}
		}
	public:
		data_tensor_type   data;
		labels_tensor_type labels;
	};

} // namespace ann

#endif
