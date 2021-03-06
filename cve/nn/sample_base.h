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

#ifndef __NN_SAMPLE_BASE_H__
#define __NN_SAMPLE_BASE_H__

#include <algorithm>
#include <random>
#include "../core/core.h"

namespace nn
{
	// Class template sample_base
	template <class Allocator = ::core::allocator<void> >
	class sample_base
	{
	public:
		// types:

		typedef typename Allocator::template rebind<size_t>::other allocator_type;
		typedef ::std::allocator_traits<allocator_type>            allocator_traits_type;
		typedef ::core::vector<size_t, allocator_type>             vector_type;
		typedef typename vector_type::const_iterator               vector_iterator_type;

		typedef typename allocator_traits_type::value_type         value_type;
		typedef typename allocator_traits_type::pointer            pointer;
		typedef typename allocator_traits_type::const_pointer      const_pointer;
		typedef typename allocator_traits_type::size_type          size_type;
		typedef typename allocator_traits_type::difference_type    difference_type;

		// construct/copy/destroy:

		sample_base(const Allocator& alloc = Allocator())
			: index(alloc)
		{}

		sample_base(size_type count)
			: index()
		{
			assign(count);
		}

		virtual void assign(size_type count)
		{
			constexpr size_t dimension = 1;
			index.assign(count, dimension);
		}

		// capacity:

		bool empty(void) const noexcept
		{
			return index.empty();
		}

		size_type size(void) const noexcept
		{
			return index.size();
		}

		// Initialization of index
		void initialize(void)
		{
			core::cpu_linear(index, size_t(0), size_t(1));
			iterator = index.cbegin();
		}

		// Random rearrangement of index
		void shuffle(unsigned int seed = 1U)
		{
			::std::default_random_engine engine(seed);
			::std::shuffle(index.data(), index.data() + index.size(), engine);
		}

		// Return the index of the next sample
		size_type next(void)
		{
			if (index.empty())
				throw ::std::domain_error(::core::vector_not_initialized);
			if (iterator == index.cend())
				iterator = index.cbegin();
			return *iterator++;
		}
	protected:
		vector_type          index;
		vector_iterator_type iterator;
	};

} // namespace nn

#endif
