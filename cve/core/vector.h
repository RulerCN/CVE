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

#ifndef __CORE_VECTOR_H__
#define __CORE_VECTOR_H__

#include "definition.h"
#include "allocator.h"
#include "uninitialized.h"
#include "scalar.h"

namespace core
{
	template <class T, class Allocator> class vector;

	// Specialize for void
	template <class Allocator>
	class vector<void, Allocator>
	{
	public:
		// types:

		using allocator_type        = Allocator;
		using allocator_traits_type = ::std::allocator_traits<allocator_type>;
		using value_type            = typename allocator_traits_type::value_type;
		using pointer               = typename allocator_traits_type::pointer;
		using const_pointer         = typename allocator_traits_type::const_pointer;
		using size_type             = typename allocator_traits_type::size_type;
		using difference_type       = typename allocator_traits_type::difference_type;

		template <class U>
		struct rebind
		{
			using other = vector<U, Allocator>;
		};
	};

	// Class template vector_type_traits

	template <class Vector, bool IsConst>
	struct vector_type_traits
	{
		using value_type      = typename Vector::value_type;
		using pointer         = typename Vector::pointer;
		using size_type       = typename Vector::size_type;
		using difference_type = typename Vector::difference_type;
		using reference       = value_type&;
	};

	template <class Vector>
	struct vector_type_traits<Vector, true>
	{
		using value_type      = typename Vector::value_type;
		using pointer         = typename Vector::const_pointer;
		using size_type       = typename Vector::size_type;
		using difference_type = typename Vector::difference_type;
		using reference       = const value_type&;
	};

	// Class template vector_iterator
	template <class Vector, bool IsConst>
	class vector_iterator
	{
	public:
		// types:

		using value_type        = typename vector_type_traits<Vector, IsConst>::value_type;
		using pointer           = typename vector_type_traits<Vector, IsConst>::pointer;
		using reference         = typename vector_type_traits<Vector, IsConst>::reference;
		using size_type         = typename vector_type_traits<Vector, IsConst>::size_type;
		using difference_type   = typename vector_type_traits<Vector, IsConst>::difference_type;

		using iterator_type     = vector_iterator<Vector, IsConst>;
		using iterator_category = ::std::bidirectional_iterator_tag;

		// construct/copy/destroy:

		vector_iterator(void) noexcept
			: step(0)
			, ptr(nullptr)
		{}
		explicit vector_iterator(pointer p, size_type stride) noexcept
			: step(stride)
			, ptr(p)
		{}
		vector_iterator(const vector_iterator<Vector, IsConst>& other) noexcept
			: step(other.step)
			, ptr(other.ptr)
		{}

		vector_iterator<Vector, IsConst>& operator=(const vector_iterator<Vector, IsConst>& other) noexcept
		{
			if (this != &other)
			{
				step = other.step;
				ptr = other.ptr;
			}
			return (*this);
		}

		operator vector_iterator<Vector, true>(void) const noexcept
		{
			return vector_iterator<Vector, true>(ptr, step);
		}

		// vector_iterator operations:

		reference operator[](size_type i) const noexcept
		{
			return static_cast<reference>(ptr[i]);
		}

		reference operator*(void) const noexcept
		{
			return static_cast<reference>(*ptr);
		}

		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		size_type stride(void) const noexcept
		{
			return step;
		}

		// increment / decrement

		vector_iterator<Vector, IsConst>& operator++(void) noexcept
		{
			ptr += step;
			return *this;
		}
		vector_iterator<Vector, IsConst>& operator--(void) noexcept
		{
			ptr -= step;
			return *this;
		}
		vector_iterator<Vector, IsConst> operator++(int) noexcept
		{
			vector_iterator tmp(*this);
			++(*this);
			return tmp;
		}
		vector_iterator<Vector, IsConst> operator--(int) noexcept
		{
			vector_iterator tmp(*this);
			--(*this);
			return tmp;
		}
		vector_iterator<Vector, IsConst>& operator+=(difference_type n) noexcept
		{
			ptr += (n * step);
			return *this;
		}
		vector_iterator<Vector, IsConst>& operator-=(difference_type n) noexcept
		{
			ptr -= (n * step);
			return *this;
		}

		// relational operators:

		template <bool b>
		bool operator==(const vector_iterator<Vector, b>& rhs) const noexcept
		{
			return (ptr == rhs.operator->());
		}
		template <bool b>
		bool operator!=(const vector_iterator<Vector, b>& rhs) const noexcept
		{
			return (ptr != rhs.operator->());
		}
	private:
		size_type step;
		pointer   ptr;
	};

	// Class template vector
	template <class T, class Allocator = allocator<T> >
	class vector : public Allocator
	{
	public:
		// types:

		using allocator_type         = Allocator;
		using scalar_type            = core::scalar<T, Allocator>;
		using const_scalar_type      = const scalar_type;
		using allocator_traits_type  = ::std::allocator_traits<allocator_type>;
		using value_type             = typename allocator_traits_type::value_type;
		using pointer                = typename allocator_traits_type::pointer;
		using const_pointer          = typename allocator_traits_type::const_pointer;
		using size_type              = typename allocator_traits_type::size_type;
		using difference_type        = typename allocator_traits_type::difference_type;

		using iterator               = vector_iterator<vector<T, Allocator>, false>;
		using const_iterator         = vector_iterator<vector<T, Allocator>, true>;
		using reverse_iterator       = ::std::reverse_iterator<iterator>;
		using const_reverse_iterator = ::std::reverse_iterator<const_iterator>;

		template <class U>
		struct rebind
		{
			using other = vector<U, Allocator>;
		};

		// construct/copy/destroy:

		explicit vector(const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{}
		explicit vector(size_type length, size_type dimension)
			: Allocator()
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(length, dimension);
		}
		vector(size_type length, size_type dimension, const value_type& value, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(length, dimension, value);
		}
		vector(size_type length, size_type dimension, const_pointer p, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(length, dimension, p);
		}
		vector(size_type length, size_type dimension, pointer p, bool copy_data, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(length, dimension, p, copy_data);
		}
		template <class InputIterator>
		vector(size_type length, size_type dimension, InputIterator first, InputIterator last, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(length, dimension, last, last);
		}
		vector(const vector<T, Allocator>& other)
			: Allocator(other.get_allocator())
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, deep_copy);
		}
		vector(vector<T, Allocator>& other, copy_mode_type copy_mode)
			: Allocator(other.get_allocator())
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, copy_mode);
		}
		vector(vector<T, Allocator>&& other)
			: Allocator(other.get_allocator())
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<vector<T, Allocator> >(other));
		}
		vector(const vector<T, Allocator>& other, const Allocator& alloc, copy_mode_type copy_mode = deep_copy)
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, copy_mode);
		}
		vector(vector<T, Allocator>& other, const Allocator& alloc, copy_mode_type copy_mode = deep_copy)
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, copy_mode);
		}
		vector(vector<T, Allocator>&& other, const Allocator& alloc)
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<vector<T, Allocator> >(other));
		}
		vector(size_type length, size_type dimension, ::std::initializer_list<T> il, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, number(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(length, dimension, il);
		}
		~vector(void)
		{
			clear();
		}
		vector<T, Allocator>& operator=(const vector<T, Allocator>& other)
		{
			if (this != &other)
			{
				clear();
				assign(other, deep_copy);
			}
			return (*this);
		}
		vector<T, Allocator>& operator=(vector<T, Allocator>&& other)
		{
			if (this != &other)
				assign(::std::forward<vector<T, Allocator> >(other));
			return (*this);
		}

		void assign(size_type length, size_type dimension)
		{
			if (!empty())
				throw ::std::domain_error(vector_is_initialized);
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			buffer = this->allocate(count);
			core::uninitialized_default_construct_n(buffer, count);
		}

		void assign(size_type length, size_type dimension, const value_type& value)
		{
			if (!empty())
				throw ::std::domain_error(vector_is_initialized);
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			buffer = this->allocate(count);
			::std::uninitialized_fill_n(buffer, count, value);
		}

		void assign(size_type length, size_type dimension, const_pointer p)
		{
			if (!empty())
				throw ::std::domain_error(vector_is_initialized);
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			buffer = this->allocate(count);
			::std::uninitialized_copy(p, p + count, buffer);
		}

		void assign(size_type length, size_type dimension, pointer p, bool copy_data)
		{
			if (!empty())
				throw ::std::domain_error(vector_is_initialized);
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			owner = copy_data;
			channels = dimension;
			number = length;
			count = number * channels;
			if (copy_data)
			{
				buffer = this->allocate(count);
				::std::uninitialized_copy(p, p + count, buffer);
			}
			else
				buffer = p;
		}

		template <class InputIterator>
		void assign(size_type length, size_type dimension, InputIterator first, InputIterator last)
		{
			if (!empty())
				throw ::std::domain_error(vector_is_initialized);
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			if (::std::distance(first, last) != length * dimension)
				throw ::std::invalid_argument(invalid_initializer_list);
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			buffer = this->allocate(count);
			::std::uninitialized_copy(first, last, buffer);
		}

		void assign(size_type length, size_type dimension, ::std::initializer_list<T> il)
		{
			assign(length, dimension, il.begin(), il.end());
		}

		void assign(const vector<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (!empty())
				throw ::std::domain_error(vector_is_initialized);
			if (other.empty())
				throw ::std::domain_error(vector_not_initialized);
			if (copy_mode == shallow_copy)
				throw ::std::invalid_argument(invalid_copy_mode);
			owner = true;
			channels = other.channels;
			number = other.number;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
				break;
			case deep_copy:
				buffer = this->allocate(count);
				::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				break;
			}
		}

		void assign(vector<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (!empty())
				throw ::std::domain_error(vector_is_initialized);
			if (other.empty())
				throw ::std::domain_error(vector_not_initialized);
			channels = other.channels;
			number = other.number;
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
				buffer = other.data();
				break;
			case deep_copy:
				owner = true;
				buffer = this->allocate(count);
				::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				break;
			}
		}

		void assign(vector<T, Allocator>&& other)
		{
			assign_rv(::std::forward<vector<T, Allocator> >(other), typename allocator_traits_type::propagate_on_container_move_assignment());
		}

		void reassign(size_type length, size_type dimension)
		{
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			if (!original_owner || buffer == nullptr)
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

		void reassign(size_type length, size_type dimension, const value_type& value)
		{
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			if (!original_owner || buffer == nullptr)
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

		void reassign(size_type length, size_type dimension, const_pointer p)
		{
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			if (!original_owner || buffer == nullptr)
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

		void reassign(size_type length, size_type dimension, pointer p, bool copy_data)
		{
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = copy_data;
			channels = dimension;
			number = length;
			count = number * channels;
			if (copy_data)
			{
				if (!original_owner || buffer == nullptr)
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
				if (original_owner && buffer != nullptr)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
				}
				buffer = p;
			}
		}

		template <class InputIterator>
		void reassign(size_type length, size_type dimension, InputIterator first, InputIterator last)
		{
			if (length == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_vector_size);
			if (::std::distance(first, last) != length * dimension)
				throw ::std::invalid_argument(invalid_initializer_list);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = dimension;
			number = length;
			count = number * channels;
			if (!original_owner || buffer == nullptr)
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

		void reassign(size_type length, size_type dimension, ::std::initializer_list<T> il)
		{
			reassign(length, dimension, il.begin(), il.end());
		}

		void reassign(const vector<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (other.empty())
				throw ::std::domain_error(vector_not_initialized);
			if (copy_mode == shallow_copy)
				throw ::std::invalid_argument(invalid_copy_mode);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			channels = other.channels;
			number = other.number;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				if (!original_owner || buffer == nullptr)
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
			case deep_copy:
				if (!original_owner || buffer == nullptr)
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

		void reassign(vector<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (other.empty())
				throw ::std::domain_error(vector_not_initialized);
			size_type original_owner = owner;
			size_type original_count = count;
			channels = other.channels;
			number = other.number;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				owner = true;
				if (!original_owner || buffer == nullptr)
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
				if (original_owner && buffer != nullptr)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
				}
				buffer = other.data();
				break;
			case deep_copy:
				owner = true;
				if (!original_owner || buffer == nullptr)
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

		// capacity:

		bool empty(void) const noexcept
		{
			return (buffer == nullptr);
		}

		size_type dimension(void) const noexcept
		{
			return channels;
		}

		size_type length(void) const noexcept
		{
			return number;
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

		scalar_type operator[](size_type i) noexcept
		{
			return scalar_type(channels, buffer + i * channels, false);
		}
		const_scalar_type operator[](size_type i) const noexcept
		{
			return const_scalar_type(channels, buffer + i * channels, false);
		}

		scalar_type at(size_type i)
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			if (i >= number)
				throw ::std::out_of_range(vector_out_of_range);
			return scalar_type(channels, buffer + i * channels, false);
		}
		const_scalar_type at(size_type i) const
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			if (i >= number)
				throw ::std::out_of_range(vector_out_of_range);
			return const_scalar_type(channels, buffer + i * channels, false);
		}

		pointer data(void) noexcept
		{
			return static_cast<pointer>(buffer);
		}
		const_pointer data(void) const noexcept
		{
			return static_cast<const_pointer>(buffer);
		}
		pointer data(size_type pos) noexcept
		{
			return static_cast<pointer>(buffer + pos * channels);
		}
		const_pointer data(size_type pos) const noexcept
		{
			return static_cast<const_pointer>(buffer + pos * channels);
		}
		pointer data(size_type pos, size_type dim) noexcept
		{
			return static_cast<pointer>(buffer + pos * channels + dim);
		}
		const_pointer data(size_type pos, size_type dim) const noexcept
		{
			return static_cast<const_pointer>(buffer + pos * channels + dim);
		}

		// modifiers:

		void fill(const value_type& value)
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			::std::fill_n(buffer, count, value);
		}

		void fill_n(size_type n, const value_type& value)
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			if (n == 0 || n > count)
				throw ::std::invalid_argument(invalid_length);
			::std::fill_n(buffer, n, value);
		}

		template <class InputIterator>
		void fill(InputIterator first, InputIterator last)
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			if (static_cast<size_type>(::std::distance(first, last)) != count)
				throw ::std::invalid_argument(invalid_iterator_distance);
			::std::copy(first, last, buffer);
		}

		void fill(::std::initializer_list<T> il)
		{
			fill(il.begin(), il.end());
		}

		void fill(const scalar_type& scalar)
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			if (scalar.empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (scalar.size() != channels)
				throw ::std::invalid_argument(invalid_dimension);
			pointer current = buffer;
			const_pointer first = scalar.data();
			const_pointer last = first + scalar.size();
			for (size_type i = 0; i < number; ++i)
			{
				::std::copy(first, last, current);
				current += channels;
			}
		}

		void fill(const vector<T, Allocator>& other)
		{
			if (empty() || other.empty())
				throw ::std::domain_error(vector_not_initialized);
			if (count != other.size())
				throw ::std::invalid_argument(invalid_size);
			::std::copy(other.buffer, other.buffer + count, buffer);
		}

		template<class Generator>
		void generate(Generator g)
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			for (size_type i = 0; i < count; ++i)
				buffer[i] = g();
		}

		void shape(size_type length, size_type dimension)
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			if (length * dimension != count)
				throw ::std::invalid_argument(invalid_vector_size);
			channels = dimension;
			number = length;
			count = number * channels;
		}

		vector<T, Allocator> reshape(size_type length, size_type dimension) const
		{
			if (empty())
				throw ::std::domain_error(vector_not_initialized);
			if (length * dimension != count)
				throw ::std::invalid_argument(invalid_vector_size);
			return vector<T, Allocator>(length, dimension, buffer);
		}

		void swap(vector<T, Allocator>& rhs) noexcept
		{
			if (this != &rhs)
			{
				::std::swap(owner, rhs.owner);
				::std::swap(channels, rhs.channels);
				::std::swap(number, rhs.number);
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
			number = 0;
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
		friend bool operator<(const vector<T, Allocator>& lhs, const vector<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(vector_different_size);
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
		friend bool operator>(const vector<T, Allocator>& lhs, const vector<T, A>& rhs)
		{
			return (rhs < lhs);
		}
		template <class A>
		friend bool operator<=(const vector<T, Allocator>& lhs, const vector<T, A>& rhs)
		{
			return !(lhs > rhs);
		}
		template <class A>
		friend bool operator>=(const vector<T, Allocator>& lhs, const vector<T, A>& rhs)
		{
			return !(lhs < rhs);
		}

		template <class A>
		friend bool operator==(const vector<T, Allocator>& lhs, const vector<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(vector_different_size);
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
		friend bool operator!=(const vector<T, Allocator>& lhs, const vector<T, A>& rhs)
		{
			return !(lhs == rhs);
		}
	private:
		void assign_rv(vector<T, Allocator>&& right, ::std::true_type)
		{
			swap(right);
		}
		void assign_rv(vector<T, Allocator>&& right, ::std::false_type)
		{
			if (get_allocator() == right.get_allocator())
				assign_rv(::std::forward<vector<T, Allocator> >(right), ::std::true_type());
			else
				assign(right, deep_copy);
		}
	private:
		bool       owner;
		size_type  channels;
		size_type  number;
		size_type  count;
		pointer    buffer;
	};

} // namespace core

#endif
