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

#ifndef __CORE_TENSOR_H__
#define __CORE_TENSOR_H__

#include "definition.h"
#include "allocator.h"
#include "uninitialized.h"
#include "scalar.h"
#include "vector.h"
#include "matrix.h"

namespace core
{
	template <class T, class Allocator> class tensor;

	// Specialize for void
	template <class Allocator>
	class tensor<void, Allocator>
	{
	public:
		// types:

		typedef Allocator                                       allocator_type;
		typedef ::std::allocator_traits<allocator_type>         allocator_traits_type;
		typedef typename allocator_traits_type::value_type      value_type;
		typedef typename allocator_traits_type::pointer         pointer;
		typedef typename allocator_traits_type::const_pointer   const_pointer;
		typedef typename allocator_traits_type::size_type       size_type;
		typedef typename allocator_traits_type::difference_type difference_type;

		template <class U> struct rebind
		{
			typedef tensor<U, Allocator> other;
		};
	};

	// Class template tensor_type_traits

	template <class Tensor, bool is_const>
	struct tensor_type_traits
	{
		typedef typename Tensor::value_type      value_type;
		typedef typename Tensor::pointer         pointer;
		typedef typename Tensor::reference       reference;
		typedef typename Tensor::size_type       size_type;
		typedef typename Tensor::difference_type difference_type;
	};

	template <class Tensor>
	struct tensor_type_traits<Tensor, true>
	{
		typedef typename Tensor::value_type      value_type;
		typedef typename Tensor::const_pointer   pointer;
		typedef typename Tensor::const_reference reference;
		typedef typename Tensor::size_type       size_type;
		typedef typename Tensor::difference_type difference_type;
	};

	// Class template tensor_iterator
	template <class Tensor, bool is_const>
	class tensor_iterator
	{
	public:
		// types:

		typedef tensor_iterator<Tensor, is_const>                              iterator_type;
		typedef ::std::bidirectional_iterator_tag                              iterator_category;

		typedef typename tensor_type_traits<Tensor, is_const>::value_type      value_type;
		typedef typename tensor_type_traits<Tensor, is_const>::pointer         pointer;
		typedef typename tensor_type_traits<Tensor, is_const>::reference       reference;
		typedef typename tensor_type_traits<Tensor, is_const>::size_type       size_type;
		typedef typename tensor_type_traits<Tensor, is_const>::difference_type difference_type;

		// construct/copy/destroy:

		tensor_iterator(void) noexcept
			: step(0)
			, ptr(nullptr)
		{}
		explicit tensor_iterator(pointer p, size_type stride) noexcept
			: step(stride)
			, ptr(p)
		{}
		tensor_iterator(const tensor_iterator<Tensor, is_const>& other) noexcept
			: step(other.step)
			, ptr(other.ptr)
		{}

		tensor_iterator<Tensor, is_const>& operator=(const tensor_iterator<Tensor, is_const>& other) noexcept
		{
			if (this != &other)
			{
				step = other.step;
				ptr = other.ptr;
			}
			return (*this);
		}

		operator tensor_iterator<Tensor, true>(void) const noexcept
		{
			return tensor_iterator<Tensor, true>(ptr, step);
		}

		// tensor_iterator operations:

		reference operator*(void) const noexcept
		{
			return static_cast<reference>(*ptr);
		}

		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		tensor_iterator<Tensor, is_const>& operator++(void) noexcept
		{
			ptr += step;
			return *this;
		}
		tensor_iterator<Tensor, is_const>& operator--(void) noexcept
		{
			ptr -= step;
			return *this;
		}
		tensor_iterator<Tensor, is_const> operator++(int) noexcept
		{
			tensor_iterator tmp(*this);
			++(*this);
			return tmp;
		}
		tensor_iterator<Tensor, is_const> operator--(int) noexcept
		{
			tensor_iterator tmp(*this);
			--(*this);
			return tmp;
		}
		tensor_iterator<Tensor, is_const>& operator+=(difference_type n) noexcept
		{
			ptr += (n * step);
			return *this;
		}
		tensor_iterator<Tensor, is_const>& operator-=(difference_type n) noexcept
		{
			ptr -= (n * step);
			return *this;
		}

		// relational operators:

		template <bool b>
		bool operator==(const tensor_iterator<Tensor, b>& rhs) const noexcept
		{
			return (ptr == rhs.operator->());
		}
		template <bool b>
		bool operator!=(const tensor_iterator<Tensor, b>& rhs) const noexcept
		{
			return (ptr != rhs.operator->());
		}
	private:
		size_type step;
		pointer   ptr;
	};

	// Class template tensor
	template <class T, class Allocator = allocator<T> >
	class tensor : public Allocator
	{
	public:
		// types:

		typedef Allocator                                       allocator_type;
		typedef ::std::allocator_traits<allocator_type>         allocator_traits_type;
		typedef core::scalar<T, Allocator>                      scalar_type;
		typedef const scalar_type                               const_scalar_type;
		typedef core::vector<T, Allocator>                      vector_type;
		typedef const vector_type                               const_vector_type;
		typedef core::matrix<T, Allocator>                      matrix_type;
		typedef const matrix_type                               const_matrix_type;

		typedef typename allocator_traits_type::value_type      value_type;
		typedef typename allocator_traits_type::pointer         pointer;
		typedef typename allocator_traits_type::const_pointer   const_pointer;
		typedef typename allocator_type::reference              reference;
		typedef typename allocator_type::const_reference        const_reference;
		typedef typename allocator_traits_type::size_type       size_type;
		typedef typename allocator_traits_type::difference_type difference_type;

		typedef tensor_iterator<tensor<T, Allocator>, false>    iterator;
		typedef tensor_iterator<tensor<T, Allocator>, true>     const_iterator;
		typedef ::std::reverse_iterator<iterator>               reverse_iterator;
		typedef ::std::reverse_iterator<const_iterator>         const_reverse_iterator;

		template <class U> struct rebind
		{
			typedef tensor<U, Allocator> other;
		};

		// construct/copy/destroy:

		explicit tensor(const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{}
		explicit tensor(size_type batch, size_type rows, size_type columns, size_type dimension)
			: Allocator()
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(batch, rows, columns, dimension);
		}
		tensor(size_type batch, size_type rows, size_type columns, size_type dimension, const value_type& value, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(batch, rows, columns, dimension, value);
		}
		tensor(size_type batch, size_type rows, size_type columns, size_type dimension, const_pointer p, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(batch, rows, columns, dimension, p);
		}
		tensor(size_type batch, size_type rows, size_type columns, size_type dimension, pointer p, bool copy_data, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(batch, rows, columns, dimension, p, copy_data);
		}
		template <class InputIterator>
		tensor(size_type batch, size_type rows, size_type columns, size_type dimension, InputIterator first, InputIterator last, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(batch, rows, columns, dimension, last, last);
		}
		tensor(const tensor<T, Allocator>& other, copy_mode_type copy_mode = deep_copy)
			: Allocator(other.get_allocator())
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, copy_mode);
		}
		tensor(tensor<T, Allocator>&& other)
			: Allocator(other.get_allocator())
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<tensor<T, Allocator> >(other));
		}
		tensor(const tensor<T, Allocator>& other, const Allocator& alloc, copy_mode_type copy_mode = deep_copy)
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, copy_mode);
		}
		tensor(tensor<T, Allocator>&& other, const Allocator& alloc)
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<tensor<T, Allocator> >(other));
		}
		tensor(size_type batch, size_type rows, size_type columns, size_type dimension, ::std::initializer_list<T> il, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, depth(0)
			, stride(0)
			, plane(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(batch, rows, columns, dimension, il);
		}
		~tensor(void)
		{
			clear();
		}
		tensor<T, Allocator>& operator=(const tensor<T, Allocator>& other)
		{
			if (this != &other)
			{
				clear();
				assign(other, other.owner ? deep_copy : shallow_copy);
			}
			return (*this);
		}
		tensor<T, Allocator>& operator=(tensor<T, Allocator>&& other)
		{
			if (this != &other)
				assign(::std::forward<tensor<T, Allocator> >(other));
			return (*this);
		}

		void assign(size_type batch, size_type rows, size_type columns, size_type dimension)
		{
			if (!empty())
				throw ::std::domain_error(tensor_is_initialized);
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			buffer = this->allocate(count);
			core::uninitialized_default_construct_n(buffer, count);
		}

		void assign(size_type batch, size_type rows, size_type columns, size_type dimension, const value_type& value)
		{
			if (!empty())
				throw ::std::domain_error(tensor_is_initialized);
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			buffer = this->allocate(count);
			::std::uninitialized_fill_n(buffer, count, value);
		}

		void assign(size_type batch, size_type rows, size_type columns, size_type dimension, const_pointer p)
		{
			if (!empty() && owner == true)
				throw ::std::domain_error(tensor_is_initialized);
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			buffer = this->allocate(count);
			::std::uninitialized_copy(p, p + count, buffer);
		}

		void assign(size_type batch, size_type rows, size_type columns, size_type dimension, pointer p, bool copy_data)
		{
			if (!empty() && owner == true)
				throw ::std::domain_error(tensor_is_initialized);
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			if (copy_data)
			{
				owner = true;
				buffer = this->allocate(count);
				::std::uninitialized_copy(p, p + count, buffer);
			}
			else
			{
				owner = false;
				buffer = p;
			}
		}

		template <class InputIterator>
		void assign(size_type batch, size_type rows, size_type columns, size_type dimension, InputIterator first, InputIterator last)
		{
			if (!empty())
				throw ::std::domain_error(tensor_is_initialized);
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			if (::std::distance(first, last) != batch * rows * columns * dimension)
				throw ::std::invalid_argument(invalid_initializer_list);
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			buffer = this->allocate(count);
			::std::uninitialized_copy(first, last, buffer);
		}

		void assign(size_type batch, size_type rows, size_type columns, size_type dimension, ::std::initializer_list<T> il)
		{
			assign(batch, rows, columns, dimension, il.begin(), il.end());
		}

		void assign(const tensor<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (!empty())
				throw ::std::domain_error(tensor_is_initialized);
			if (other.empty())
				throw ::std::domain_error(tensor_not_initialized);
			channels = other.channels;
			width = other.width;
			height = other.height;
			depth = other.depth;
			stride = other.stride;
			plane = other.plane;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				owner = true;
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
				break;
			case shallow_copy:
				owner = false;
				buffer = const_cast<pointer>(other.data());
				break;
			case deep_copy:
				owner = true;
				buffer = this->allocate(count);
				::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				break;
			}
		}

		void assign(tensor<T, Allocator>&& other)
		{
			assign_rv(std::forward<tensor<T, Allocator> >(other), typename allocator_type::propagate_on_container_move_assignment());
		}

		void reassign(size_type batch, size_type rows, size_type columns, size_type dimension)
		{
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			if (buffer == nullptr || !original_owner)
			{
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
			}
		}

		void reassign(size_type batch, size_type rows, size_type columns, size_type dimension, const value_type& value)
		{
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			if (buffer == nullptr || !original_owner)
			{
				buffer = this->allocate(count);
				::std::uninitialized_fill_n(buffer, count, value);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				::std::uninitialized_fill_n(buffer, count, value);
			}
			else
				::std::fill_n(buffer, count, value);
		}

		void reassign(size_type batch, size_type rows, size_type columns, size_type dimension, const_pointer p)
		{
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			if (buffer == nullptr || !original_owner)
			{
				buffer = this->allocate(count);
				::std::uninitialized_copy(p, p + count, buffer);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				::std::uninitialized_copy(p, p + count, buffer);
			}
			else
				::std::copy(p, p + count, buffer);
		}

		void reassign(size_type batch, size_type rows, size_type columns, size_type dimension, pointer p, bool copy_data)
		{
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = copy_data;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			if (copy_data)
			{
				if (buffer == nullptr || !original_owner)
				{
					buffer = this->allocate(count);
					::std::uninitialized_copy(p, p + count, buffer);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					::std::uninitialized_copy(p, p + count, buffer);
				}
				else
					::std::copy(p, p + count, buffer);
			}
			else
			{
				if (buffer != nullptr)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
				}
				buffer = p;
			}
		}

		template <class InputIterator>
		void reassign(size_type batch, size_type rows, size_type columns, size_type dimension, InputIterator first, InputIterator last)
		{
			if (batch == 0 || rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_tensor_size);
			if (::std::distance(first, last) != batch * rows * columns * dimension)
				throw ::std::invalid_argument(invalid_initializer_list);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
			count = depth * plane;
			if (buffer == nullptr || !original_owner)
			{
				buffer = this->allocate(count);
				::std::uninitialized_copy(first, last, buffer);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				::std::uninitialized_copy(first, last, buffer);
			}
			else
				::std::copy(first, last, buffer);
		}

		void reassign(size_type batch, size_type rows, size_type columns, size_type dimension, ::std::initializer_list<T> il)
		{
			reassign(batch, rows, columns, dimension, il.begin(), il.end());
		}

		void reassign(const tensor<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (other.empty())
				throw ::std::domain_error(tensor_not_initialized);
			size_type original_owner = owner;
			size_type original_count = count;
			channels = other.channels;
			width = other.width;
			height = other.height;
			depth = other.depth;
			stride = other.stride;
			plane = other.plane;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				owner = true;
				if (buffer == nullptr || !original_owner)
				{
					buffer = this->allocate(count);
					core::uninitialized_default_construct_n(buffer, count);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					core::uninitialized_default_construct_n(buffer, count);
				}
				break;
			case shallow_copy:
				owner = false;
				if (buffer != nullptr)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
				}
				buffer = const_cast<pointer>(other.data());
				break;
			case deep_copy:
				owner = true;
				if (buffer == nullptr || !original_owner)
				{
					buffer = this->allocate(count);
					::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				}
				else
					::std::copy(other.buffer, other.buffer + count, buffer);
				break;
			}
		}

		// iterators:

		iterator begin(void) noexcept
		{
			return iterator(buffer, channels);
		}
		const_iterator begin(void) const noexcept
		{
			return const_iterator(buffer, channels);
		}
		const_iterator cbegin(void) const noexcept
		{
			return const_iterator(buffer, channels);
		}
		iterator end(void) noexcept
		{
			return iterator(buffer + count, channels);
		}
		const_iterator end(void) const noexcept
		{
			return const_iterator(buffer + count, channels);
		}
		const_iterator cend(void) const noexcept
		{
			return const_iterator(buffer + count, channels);
		}
		reverse_iterator rbegin(void) noexcept
		{
			return reverse_iterator(end());
		}
		const_reverse_iterator rbegin(void) const noexcept
		{
			return const_reverse_iterator(end());
		}
		const_reverse_iterator crbegin(void) const noexcept
		{
			return const_reverse_iterator(cend());
		}
		reverse_iterator rend(void) noexcept
		{
			return reverse_iterator(begin());
		}
		const_reverse_iterator rend(void) const noexcept
		{
			return const_reverse_iterator(begin());
		}
		const_reverse_iterator crend(void) const noexcept
		{
			return const_reverse_iterator(cbegin());
		}

		iterator mbegin(void) noexcept
		{
			return iterator(buffer, plane);
		}
		const_iterator mbegin(void) const noexcept
		{
			return const_iterator(buffer, plane);
		}
		const_iterator cmbegin(void) const noexcept
		{
			return const_iterator(buffer, plane);
		}
		iterator mend(void) noexcept
		{
			return iterator(buffer + count, plane);
		}
		const_iterator mend(void) const noexcept
		{
			return const_iterator(buffer + count, plane);
		}
		const_iterator cmend(void) const noexcept
		{
			return const_iterator(buffer + count, plane);
		}
		reverse_iterator rmbegin(void) noexcept
		{
			return reverse_iterator(mend());
		}
		const_reverse_iterator rmbegin(void) const noexcept
		{
			return const_reverse_iterator(mend());
		}
		const_reverse_iterator crmbegin(void) const noexcept
		{
			return const_reverse_iterator(cmend());
		}
		reverse_iterator rmend(void) noexcept
		{
			return reverse_iterator(mbegin());
		}
		const_reverse_iterator rmend(void) const noexcept
		{
			return const_reverse_iterator(mbegin());
		}
		const_reverse_iterator crmend(void) const noexcept
		{
			return const_reverse_iterator(cmbegin());
		}

		iterator vbegin(iterator it) noexcept
		{
			return iterator(it.operator->(), stride);
		}
		iterator vbegin(reverse_iterator it) noexcept
		{
			return iterator(it.operator->(), stride);
		}
		const_iterator vbegin(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		const_iterator vbegin(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		const_iterator vbegin(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		const_iterator vbegin(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		const_iterator cvbegin(iterator it) noexcept
		{
			return const_iterator(it.operator->(), stride);
		}
		const_iterator cvbegin(reverse_iterator it) noexcept
		{
			return const_iterator(it.operator->(), stride);
		}
		const_iterator cvbegin(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		const_iterator cvbegin(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		const_iterator cvbegin(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		const_iterator cvbegin(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), stride);
		}
		iterator vend(iterator it) noexcept
		{
			return iterator(it.operator->() + plane, stride);
		}
		iterator vend(reverse_iterator it) noexcept
		{
			return iterator(it.operator->() + plane, stride);
		}
		const_iterator vend(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		const_iterator vend(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		const_iterator vend(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		const_iterator vend(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		const_iterator cvend(iterator it) noexcept
		{
			return const_iterator(it.operator->() + plane, stride);
		}
		const_iterator cvend(reverse_iterator it) noexcept
		{
			return const_iterator(it.operator->() + plane, stride);
		}
		const_iterator cvend(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		const_iterator cvend(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		const_iterator cvend(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		const_iterator cvend(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + plane), stride);
		}
		reverse_iterator rvbegin(iterator it) noexcept
		{
			return reverse_iterator(vend(it));
		}
		reverse_iterator rvbegin(reverse_iterator it) noexcept
		{
			return reverse_iterator(vend(it));
		}
		const_reverse_iterator rvbegin(const_iterator it) noexcept
		{
			return const_reverse_iterator(vend(it));
		}
		const_reverse_iterator rvbegin(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(vend(it));
		}
		const_reverse_iterator rvbegin(const_iterator it) const noexcept
		{
			return const_reverse_iterator(vend(it));
		}
		const_reverse_iterator rvbegin(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(vend(it));
		}
		const_reverse_iterator crvbegin(iterator it) noexcept
		{
			return const_reverse_iterator(cvend(it));
		}
		const_reverse_iterator crvbegin(reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cvend(it));
		}
		const_reverse_iterator crvbegin(const_iterator it) noexcept
		{
			return const_reverse_iterator(cvend(it));
		}
		const_reverse_iterator crvbegin(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cvend(it));
		}
		const_reverse_iterator crvbegin(const_iterator it) const noexcept
		{
			return const_reverse_iterator(cvend(it));
		}
		const_reverse_iterator crvbegin(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(cvend(it));
		}
		reverse_iterator rvend(iterator it) noexcept
		{
			return reverse_iterator(vbegin(it));
		}
		reverse_iterator rvend(reverse_iterator it) noexcept
		{
			return reverse_iterator(vbegin(it));
		}
		const_reverse_iterator rvend(const_iterator it) noexcept
		{
			return const_reverse_iterator(vbegin(it));
		}
		const_reverse_iterator rvend(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(vbegin(it));
		}
		const_reverse_iterator rvend(const_iterator it) const noexcept
		{
			return const_reverse_iterator(vbegin(it));
		}
		const_reverse_iterator rvend(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(vbegin(it));
		}
		const_reverse_iterator crvend(iterator it) noexcept
		{
			return const_reverse_iterator(cvbegin(it));
		}
		const_reverse_iterator crvend(reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cvbegin(it));
		}
		const_reverse_iterator crvend(const_iterator it) noexcept
		{
			return const_reverse_iterator(cvbegin(it));
		}
		const_reverse_iterator crvend(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cvbegin(it));
		}
		const_reverse_iterator crvend(const_iterator it) const noexcept
		{
			return const_reverse_iterator(cvbegin(it));
		}
		const_reverse_iterator crvend(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(cvbegin(it));
		}

		iterator begin(iterator it) noexcept
		{
			return iterator(it.operator->(), channels);
		}
		iterator begin(reverse_iterator it) noexcept
		{
			return iterator(it.operator->(), channels);
		}
		const_iterator begin(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		const_iterator begin(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		const_iterator begin(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		const_iterator begin(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		const_iterator cbegin(iterator it) noexcept
		{
			return const_iterator(it.operator->(), channels);
		}
		const_iterator cbegin(reverse_iterator it) noexcept
		{
			return const_iterator(it.operator->(), channels);
		}
		const_iterator cbegin(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		const_iterator cbegin(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		const_iterator cbegin(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		const_iterator cbegin(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->()), channels);
		}
		iterator end(iterator it) noexcept
		{
			return iterator(it.operator->() + stride, channels);
		}
		iterator end(reverse_iterator it) noexcept
		{
			return iterator(it.operator->() + stride, channels);
		}
		const_iterator end(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		const_iterator end(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		const_iterator end(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		const_iterator end(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		const_iterator cend(iterator it) noexcept
		{
			return const_iterator(it.operator->() + stride, channels);
		}
		const_iterator cend(reverse_iterator it) noexcept
		{
			return const_iterator(it.operator->() + stride, channels);
		}
		const_iterator cend(const_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		const_iterator cend(const_reverse_iterator it) noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		const_iterator cend(const_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		const_iterator cend(const_reverse_iterator it) const noexcept
		{
			return const_iterator(const_cast<pointer>(it.operator->() + stride), channels);
		}
		reverse_iterator rbegin(iterator it) noexcept
		{
			return reverse_iterator(end(it));
		}
		reverse_iterator rbegin(reverse_iterator it) noexcept
		{
			return reverse_iterator(end(it));
		}
		const_reverse_iterator rbegin(const_iterator it) noexcept
		{
			return const_reverse_iterator(end(it));
		}
		const_reverse_iterator rbegin(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(end(it));
		}
		const_reverse_iterator rbegin(const_iterator it) const noexcept
		{
			return const_reverse_iterator(end(it));
		}
		const_reverse_iterator rbegin(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(end(it));
		}
		const_reverse_iterator crbegin(iterator it) noexcept
		{
			return const_reverse_iterator(cend(it));
		}
		const_reverse_iterator crbegin(reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cend(it));
		}
		const_reverse_iterator crbegin(const_iterator it) noexcept
		{
			return const_reverse_iterator(cend(it));
		}
		const_reverse_iterator crbegin(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cend(it));
		}
		const_reverse_iterator crbegin(const_iterator it) const noexcept
		{
			return const_reverse_iterator(cend(it));
		}
		const_reverse_iterator crbegin(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(cend(it));
		}
		reverse_iterator rend(iterator it) noexcept
		{
			return reverse_iterator(begin(it));
		}
		reverse_iterator rend(reverse_iterator it) noexcept
		{
			return reverse_iterator(begin(it));
		}
		const_reverse_iterator rend(const_iterator it) noexcept
		{
			return const_reverse_iterator(begin(it));
		}
		const_reverse_iterator rend(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(begin(it));
		}
		const_reverse_iterator rend(const_iterator it) const noexcept
		{
			return const_reverse_iterator(begin(it));
		}
		const_reverse_iterator rend(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(begin(it));
		}
		const_reverse_iterator crend(iterator it) noexcept
		{
			return const_reverse_iterator(cbegin(it));
		}
		const_reverse_iterator crend(reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cbegin(it));
		}
		const_reverse_iterator crend(const_iterator it) noexcept
		{
			return const_reverse_iterator(cbegin(it));
		}
		const_reverse_iterator crend(const_reverse_iterator it) noexcept
		{
			return const_reverse_iterator(cbegin(it));
		}
		const_reverse_iterator crend(const_iterator it) const noexcept
		{
			return const_reverse_iterator(cbegin(it));
		}
		const_reverse_iterator crend(const_reverse_iterator it) const noexcept
		{
			return const_reverse_iterator(cbegin(it));
		}

		// capacity:

		bool empty(void) const noexcept
		{
			return (buffer == nullptr);
		}

		size_type dimension(void) const noexcept
		{
			return channels;
		}

		size_type columns(void) const noexcept
		{
			return width;
		}

		size_type rows(void) const noexcept
		{
			return height;
		}

		size_type batch(void) const noexcept
		{
			return depth;
		}

		size_type area(void) const noexcept
		{
			return (height * width);
		}

		size_type volume(void) const noexcept
		{
			return (depth * height * width);
		}

		size_type row_size(void) const noexcept
		{
			return stride;
		}

		size_type matrix_size(void) const noexcept
		{
			return plane;
		}

		size_type size(void) const noexcept
		{
			return count;
		}

		size_type max_size(void) const noexcept
		{
			return this->max_size();
		}

		// element access:

		matrix_type operator[](size_type i) noexcept
		{
			return matrix_type(height, width, channels, buffer + i * plane, false);
		}
		const_matrix_type operator[](size_type i) const noexcept
		{
			return const_matrix_type(height, width, channels, buffer + i * plane, false);
		}

		matrix_type at(size_type i)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (i >= depth)
				throw ::std::out_of_range(tensor_out_of_range);
			return matrix_type(height, width, channels, buffer + i * plane, false);
		}
		const_matrix_type at(size_type i) const
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (i >= depth)
				throw ::std::out_of_range(tensor_out_of_range);
			return const_matrix_type(height, width, channels, buffer + i * plane, false);
		}

		pointer data(void) noexcept
		{
			return static_cast<pointer>(buffer);
		}
		const_pointer data(void) const noexcept
		{
			return static_cast<const_pointer>(buffer);
		}
		pointer data(size_type batch) noexcept
		{
			return static_cast<pointer>(buffer + batch * plane);
		}
		const_pointer data(size_type batch) const noexcept
		{
			return static_cast<const_pointer>(buffer + batch * plane);
		}
		pointer data(size_type batch, size_type row) noexcept
		{
			return static_cast<pointer>(buffer + batch * plane + row * stride);
		}
		const_pointer data(size_type batch, size_type row) const noexcept
		{
			return static_cast<const_pointer>(buffer + batch * plane + row * stride);
		}
		pointer data(size_type batch, size_type row, size_type column) noexcept
		{
			return static_cast<pointer>(buffer + batch * plane + row * stride + column * channels);
		}
		const_pointer data(size_type batch, size_type row, size_type column) const noexcept
		{
			return static_cast<const_pointer>(buffer + batch * plane + row * stride + column * channels);
		}
		pointer data(size_type batch, size_type row, size_type column, size_type dim) noexcept
		{
			return static_cast<pointer>(buffer + batch * plane + row * stride + column * channels + dim);
		}
		const_pointer data(size_type batch, size_type row, size_type column, size_type dim) const noexcept
		{
			return static_cast<const_pointer>(buffer + batch * plane + row * stride + column * channels + dim);
		}

		scalar_type scalar(iterator it) noexcept
		{
			return scalar_type(channels, it.operator->(), false);
		}
		scalar_type scalar(reverse_iterator it) noexcept
		{
			return scalar_type(channels, it.operator->(), false);
		}
		const_scalar_type scalar(const_iterator it) noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()), false);
		}
		const_scalar_type scalar(const_reverse_iterator it) noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()), false);
		}
		const_scalar_type scalar(const_iterator it) const noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()), false);
		}
		const_scalar_type scalar(const_reverse_iterator it) const noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()), false);
		}

		vector_type vector(iterator it) noexcept
		{
			return vector_type(width, channels, it.operator->(), false);
		}
		vector_type vector(reverse_iterator it) noexcept
		{
			return vector_type(width, channels, it.operator->(), false);
		}
		const_vector_type vector(const_iterator it) noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()), false);
		}
		const_vector_type vector(const_reverse_iterator it) noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()), false);
		}
		const_vector_type vector(const_iterator it) const noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()), false);
		}
		const_vector_type vector(const_reverse_iterator it) const noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()), false);
		}

		matrix_type matrix(iterator it) noexcept
		{
			return matrix_type(height, width, channels, it.operator->(), false);
		}
		matrix_type matrix(reverse_iterator it) noexcept
		{
			return matrix_type(height, width, channels, it.operator->(), false);
		}
		const_matrix_type matrix(const_iterator it) noexcept
		{
			return const_matrix_type(height, width, channels, const_cast<pointer>(it.operator->()), false);
		}
		const_matrix_type matrix(const_reverse_iterator it) noexcept
		{
			return const_matrix_type(height, width, channels, const_cast<pointer>(it.operator->()), false);
		}
		const_matrix_type matrix(const_iterator it) const noexcept
		{
			return const_matrix_type(height, width, channels, const_cast<pointer>(it.operator->()), false);
		}
		const_matrix_type matrix(const_reverse_iterator it) const noexcept
		{
			return const_matrix_type(height, width, channels, const_cast<pointer>(it.operator->()), false);
		}

		// modifiers:

		void fill(const value_type& value)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			::std::fill_n(buffer, count, value);
		}

		void fill_n(size_type n, const value_type& value)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (n == 0 || n > count)
				throw ::std::invalid_argument(invalid_length);
			::std::fill_n(buffer, n, value);
		}

		template <class InputIterator>
		void fill(InputIterator first, InputIterator last)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (static_cast<size_type>(::std::distance(first, last)) != count)
				throw ::std::invalid_argument(invalid_iterator_distance);
			::std::copy(first, last, buffer);
		}

		void fill(::std::initializer_list<T> il)
		{
			fill(il.begin(), il.end());
		}

		void fill(const tensor<T, Allocator>& other)
		{
			if (empty() || other.empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (count != other.size())
				throw ::std::invalid_argument(invalid_length);
			::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
		}

		void linear_fill(const value_type& init, const value_type& delta)
		{
			value_type value(init);
			for (size_type i = 0; i < count; ++i)
			{
				buffer[i] = value;
				value += delta;
			}
		}

		void linear_fill(const value_type& init, const value_type& mat_delta, const value_type& row_delta, const value_type& col_delta)
		{
			matrix_type first_mat = matrix_type(height, width, channels, buffer);
			first_mat.linear_fill(init, row_delta, col_delta);
			pointer current = buffer;
			for (size_type j = 1; j < depth; ++j)
			{
				pointer next = current + plane;
				for (size_type i = 0; i < plane; ++i)
					next[i] = current[i] + mat_delta;
				current = next;
			}
		}

		void value(const scalar_type& element)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (element.size() != channels)
				throw ::std::invalid_argument(invalid_dimension);
			pointer current = buffer;
			size_type number = volume();
			const_pointer first = element.data();
			const_pointer last = first + element.size();
			for (size_type i = 0; i < number; ++i)
			{
				::std::copy(first, last, current);
				current += channels;
			}
		}

		void linear_value(const scalar_type& init, const value_type& mat_delta, const value_type& row_delta, const value_type& col_delta)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (init.size() != channels)
				throw ::std::invalid_argument(invalid_dimension);
			matrix_type first_mat = matrix_type(height, width, channels, buffer);
			first_mat.linear_value(init, row_delta, col_delta);
			pointer current = buffer;
			for (size_type j = 1; j < depth; ++j)
			{
				pointer next = current + plane;
				for (size_type i = 0; i < plane; ++i)
					next[i] = current[i] + mat_delta;
				current = next;
			}
		}

		template<class Generator>
		void generate(Generator g)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			for (size_type i = 0; i < count; ++i)
				buffer[i] = g();
		}

		void reshape(size_type batch, size_type rows, size_type columns, size_type dimension)
		{
			if (empty())
				throw ::std::domain_error(tensor_not_initialized);
			if (count != batch * rows * columns * dimension)
				throw ::std::invalid_argument(invalid_tensor_size);
			channels = dimension;
			width = columns;
			height = rows;
			depth = batch;
			stride = width * channels;
			plane = height * stride;
		}

		void build(size_type batch, size_type rows, size_type columns, size_type dimension)
		{
			if (empty())
				assign(batch, rows, columns, dimension);
			else if (count != batch * rows * columns * dimension)
			{
				clear();
				assign(batch, rows, columns, dimension);
			}
			else
			{
				channels = dimension;
				width = columns;
				height = rows;
				depth = batch;
				stride = width * channels;
				plane = height * stride;
			}
		}

		void swap(tensor<T, Allocator>& rhs) noexcept
		{
			if (this != &rhs)
			{
				::std::swap(owner, rhs.owner);
				::std::swap(channels, rhs.channels);
				::std::swap(width, rhs.width);
				::std::swap(height, rhs.height);
				::std::swap(depth, rhs.depth);
				::std::swap(stride, rhs.stride);
				::std::swap(plane, rhs.plane);
				::std::swap(count, rhs.count);
				::std::swap(buffer, rhs.buffer);
			}
		}

		void clear(void) noexcept
		{
			if (owner && buffer != nullptr)
			{
				core::destroy_n(buffer, count);
				this->deallocate(buffer, count);
				buffer = nullptr;
			}
			owner = true;
			channels = 0;
			width = 0;
			height = 0;
			depth = 0;
			stride = 0;
			plane = 0;
			count = 0;
		}

		// allocator

		allocator_type get_allocator(void) const noexcept
		{
			return *static_cast<const allocator_type*>(this);
		}
	public:
		// operator:

		template <class A>
		friend bool operator<(const tensor<T, Allocator>& lhs, const tensor<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(tensor_different_size);
			bool rst = true;
			const_pointer ptr1 = lhs.data();
			const_pointer ptr2 = rhs.data();
			for (size_t i = 0; i < lhs.count; ++i)
			{
				if (!(ptr1[i] < ptr2[i]))
				{
					rst = false;
					break;
				}
			}
			return rst;
		}
		template <class A>
		friend bool operator>(const tensor<T, Allocator>& lhs, const tensor<T, A>& rhs)
		{
			return (rhs < lhs);
		}
		template <class A>
		friend bool operator<=(const tensor<T, Allocator>& lhs, const tensor<T, A>& rhs)
		{
			return !(lhs > rhs);
		}
		template <class A>
		friend bool operator>=(const tensor<T, Allocator>& lhs, const tensor<T, A>& rhs)
		{
			return !(lhs < rhs);
		}

		template <class A>
		friend bool operator==(const tensor<T, Allocator>& lhs, const tensor<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(tensor_different_size);
			bool rst = true;
			const_pointer ptr1 = lhs.data();
			const_pointer ptr2 = rhs.data();
			for (size_t i = 0; i < lhs.count; ++i)
			{
				if (!(ptr1[i] == ptr2[i]))
				{
					rst = false;
					break;
				}
			}
			return rst;
		}
		template <class A>
		friend bool operator!=(const tensor<T, Allocator>& lhs, const tensor<T, A>& rhs)
		{
			return !(lhs == rhs);
		}
	private:
		void assign_rv(tensor<T, Allocator>&& right, ::std::true_type)
		{
			swap(right);
		}
		void assign_rv(tensor<T, Allocator>&& right, ::std::false_type)
		{
			if (get_allocator() == right.get_allocator())
				assign_rv(::std::forward<tensor<T, Allocator> >(right), ::std::true_type());
			else
				assign(right, deep_copy);
		}
	private:
		bool       owner;
		size_type  channels;
		size_type  width;
		size_type  height;
		size_type  depth;
		size_type  stride;
		size_type  plane;
		size_type  count;
		pointer    buffer;
	};

} // namespace core

#endif
