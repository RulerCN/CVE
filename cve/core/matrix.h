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

#ifndef __CORE_MATRIX_H__
#define __CORE_MATRIX_H__

#include "definition.h"
#include "allocator.h"
#include "uninitialized.h"
#include "scalar.h"
#include "vector.h"

namespace core
{
	template <class T, class Allocator> class matrix;

	// Specialize for void
	template <class Allocator>
	class matrix<void, Allocator>
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
			typedef matrix<U, Allocator> other;
		};
	};

	// Class template matrix_type_traits

	template <class Matrix, bool is_const>
	struct matrix_type_traits
	{
		typedef typename Matrix::value_type      value_type;
		typedef typename Matrix::pointer         pointer;
		typedef typename Matrix::reference       reference;
		typedef typename Matrix::size_type       size_type;
		typedef typename Matrix::difference_type difference_type;
	};

	template <class Matrix>
	struct matrix_type_traits<Matrix, true>
	{
		typedef typename Matrix::value_type      value_type;
		typedef typename Matrix::const_pointer   pointer;
		typedef typename Matrix::const_reference reference;
		typedef typename Matrix::size_type       size_type;
		typedef typename Matrix::difference_type difference_type;
	};

	// Class template matrix_iterator
	template <class Matrix, bool is_const>
	class matrix_iterator
	{
	public:
		// types:

		typedef matrix_iterator<Matrix, is_const>                              iterator_type;
		typedef ::std::bidirectional_iterator_tag                              iterator_category;

		typedef typename matrix_type_traits<Matrix, is_const>::value_type      value_type;
		typedef typename matrix_type_traits<Matrix, is_const>::pointer         pointer;
		typedef typename matrix_type_traits<Matrix, is_const>::reference       reference;
		typedef typename matrix_type_traits<Matrix, is_const>::size_type       size_type;
		typedef typename matrix_type_traits<Matrix, is_const>::difference_type difference_type;

		// construct/copy/destroy:

		matrix_iterator(void) noexcept
			: step(0)
			, ptr(nullptr)
		{}
		explicit matrix_iterator(pointer p, size_type stride) noexcept
			: step(stride)
			, ptr(p)
		{}
		matrix_iterator(const matrix_iterator<Matrix, is_const>& x) noexcept
			: step(x.step)
			, ptr(x.ptr)
		{}

		matrix_iterator<Matrix, is_const>& operator=(const matrix_iterator<Matrix, is_const>& x) noexcept
		{
			if (this != &x)
			{
				step = x.step;
				ptr = x.ptr;
			}
			return (*this);
		}

		operator matrix_iterator<Matrix, true>(void) const noexcept
		{
			return matrix_iterator<Matrix, true>(ptr, step);
		}

		// matrix_iterator operations:

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

		// increment / decrement

		matrix_iterator<Matrix, is_const>& operator++(void) noexcept
		{
			ptr += step;
			return *this;
		}
		matrix_iterator<Matrix, is_const>& operator--(void) noexcept
		{
			ptr -= step;
			return *this;
		}
		matrix_iterator<Matrix, is_const> operator++(int) noexcept
		{
			matrix_iterator tmp(*this);
			++(*this);
			return tmp;
		}
		matrix_iterator<Matrix, is_const> operator--(int) noexcept
		{
			matrix_iterator tmp(*this);
			--(*this);
			return tmp;
		}
		matrix_iterator<Matrix, is_const>& operator+=(difference_type n) noexcept
		{
			ptr += (n * step);
			return *this;
		}
		matrix_iterator<Matrix, is_const>& operator-=(difference_type n) noexcept
		{
			ptr -= (n * step);
			return *this;
		}

		// relational operators:

		template <bool b>
		bool operator==(const matrix_iterator<Matrix, b>& rhs) const noexcept
		{
			return (ptr == rhs.operator->());
		}
		template <bool b>
		bool operator!=(const matrix_iterator<Matrix, b>& rhs) const noexcept
		{
			return (ptr != rhs.operator->());
		}
	private:
		size_type step;
		pointer   ptr;
	};

	// Class template matrix
	template <class T, class Allocator = allocator<T> >
	class matrix : public Allocator
	{
	public:
		// types:

		typedef Allocator                                       allocator_type;
		typedef ::std::allocator_traits<allocator_type>         allocator_traits_type;
		typedef core::scalar<T, Allocator>                      scalar_type;
		typedef const scalar_type                               const_scalar_type;
		typedef core::vector<T, Allocator>                      vector_type;
		typedef const vector_type                               const_vector_type;

		typedef typename allocator_traits_type::value_type      value_type;
		typedef typename allocator_traits_type::pointer         pointer;
		typedef typename allocator_traits_type::const_pointer   const_pointer;
		typedef typename allocator_type::reference              reference;
		typedef typename allocator_type::const_reference        const_reference;
		typedef typename allocator_traits_type::size_type       size_type;
		typedef typename allocator_traits_type::difference_type difference_type;

		typedef matrix_iterator<matrix<T, Allocator>, false>    iterator;
		typedef matrix_iterator<matrix<T, Allocator>, true>     const_iterator;
		typedef ::std::reverse_iterator<iterator>               reverse_iterator;
		typedef ::std::reverse_iterator<const_iterator>         const_reverse_iterator;

		template <class U> struct rebind
		{
			typedef matrix<U, Allocator> other;
		};

		// construct/copy/destroy:

		explicit matrix(const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{}
		explicit matrix(size_type rows, size_type columns, size_type dimension)
			: Allocator()
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(rows, columns, dimension);
		}
		matrix(size_type rows, size_type columns, size_type dimension, const value_type& value, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(rows, columns, dimension, value);
		}
		matrix(size_type rows, size_type columns, size_type dimension, pointer p, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			create(rows, columns, dimension, p);
		}
		template <class InputIterator>
		matrix(size_type columns, size_type dimension, InputIterator first, InputIterator last, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(columns, dimension, last, last);
		}
		matrix(const matrix<T, Allocator>& x)
			: Allocator(x.get_allocator())
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(x);
		}
		matrix(matrix<T, Allocator>&& x)
			: Allocator(x.get_allocator())
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<matrix<T, Allocator> >(x));
		}
		matrix(const matrix<T, Allocator>& x, const Allocator& alloc)
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(x);
		}
		matrix(matrix<T, Allocator>&& x, const Allocator& alloc)
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<matrix<T, Allocator> >(x));
		}
		matrix(size_type columns, size_type dimension, ::std::initializer_list<T> il, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, channels(0)
			, width(0)
			, height(0)
			, stride(0)
			, count(0)
			, buffer(nullptr)
		{
			assign(columns, dimension, il);
		}
		~matrix(void)
		{
			clear();
		}
		matrix<T, Allocator>& operator=(const matrix<T, Allocator>& x)
		{
			if (this != &x)
			{
				clear();
				if (x.owner)
					assign(x);
				else
					create(x);
			}
			return (*this);
		}
		matrix<T, Allocator>& operator=(matrix<T, Allocator>&& x)
		{
			if (this != &x)
				assign(::std::forward<matrix<T, Allocator> >(x));
			return (*this);
		}
		matrix<T, Allocator>& operator=(::std::initializer_list<T> il)
		{
			clear();
			assign(il);
			return (*this);
		}

		void assign(size_type rows, size_type columns, size_type dimension)
		{
			if (!empty())
				throw ::std::domain_error(matrix_is_initialized);
			if (rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_matrix_size);
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			stride = width * channels;
			count = height * stride;
			buffer = this->allocate(count);
			core::uninitialized_default_construct_n(buffer, count);
		}

		void assign(size_type rows, size_type columns, size_type dimension, const value_type& value)
		{
			if (!empty())
				throw ::std::domain_error(matrix_is_initialized);
			if (rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_matrix_size);
			owner = true;
			channels = dimension;
			width = columns;
			height = rows;
			stride = width * channels;
			count = height * stride;
			buffer = this->allocate(count);
			::std::uninitialized_fill_n(buffer, count, value);
		}

		template <class InputIterator>
		void assign(size_type columns, size_type dimension, InputIterator first, InputIterator last)
		{
			if (!empty())
				throw ::std::domain_error(matrix_is_initialized);
			if (::std::distance(first, last) <= 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_initializer_list);
			owner = true;
			channels = dimension;
			width = columns;
			stride = width * channels;
			height = (static_cast<size_type>(::std::distance(first, last)) + stride - 1) / stride;
			count = height * stride;
			buffer = this->allocate(count);
			::std::uninitialized_copy(first, last, buffer);
		}

		void assign(size_type columns, size_type dimension, ::std::initializer_list<T> il)
		{
			assign(columns, dimension, il.begin(), il.end());
		}

		void assign(const matrix<T, Allocator>& x)
		{
			if (!empty())
				throw ::std::domain_error(matrix_is_initialized);
			if (x.empty())
				throw ::std::domain_error(matrix_not_initialized);
			owner = true;
			channels = x.channels;
			width = x.width;
			height = x.height;
			stride = x.stride;
			count = x.count;
			buffer = this->allocate(count);
			::std::uninitialized_copy(x.buffer, x.buffer + count, buffer);
		}

		void assign(matrix<T, Allocator>&& x)
		{
			assign_rv(std::forward<matrix<T, Allocator> >(x), typename allocator_type::propagate_on_container_move_assignment());
		}

		void create(size_type rows, size_type columns, size_type dimension, pointer p)
		{
			if (!empty())
				throw ::std::domain_error(matrix_is_initialized);
			if (rows == 0 || columns == 0 || dimension == 0)
				throw ::std::invalid_argument(invalid_matrix_size);
			owner = false;
			channels = dimension;
			width = columns;
			height = rows;
			stride = width * channels;
			count = height * stride;
			buffer = p;
		}

		void create(matrix<T, Allocator>& x)
		{
			create(x.height, x.width, x.channels, x.buffer);
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

		iterator vbegin(void) noexcept
		{
			return iterator(buffer, stride);
		}
		const_iterator vbegin(void) const noexcept
		{
			return const_iterator(buffer, stride);
		}
		const_iterator cvbegin(void) const noexcept
		{
			return const_iterator(buffer, stride);
		}
		iterator vend(void) noexcept
		{
			return iterator(buffer + count, stride);
		}
		const_iterator vend(void) const noexcept
		{
			return const_iterator(buffer + count, stride);
		}
		const_iterator cvend(void) const noexcept
		{
			return const_iterator(buffer + count, stride);
		}
		reverse_iterator rvbegin(void) noexcept
		{
			return reverse_iterator(vend());
		}
		const_reverse_iterator rvbegin(void) const noexcept
		{
			return const_reverse_iterator(vend());
		}
		const_reverse_iterator crvbegin(void) const noexcept
		{
			return const_reverse_iterator(cvend());
		}
		reverse_iterator rvend(void) noexcept
		{
			return reverse_iterator(vbegin());
		}
		const_reverse_iterator rvend(void) const noexcept
		{
			return const_reverse_iterator(vbegin());
		}
		const_reverse_iterator crvend(void) const noexcept
		{
			return const_reverse_iterator(cvbegin());
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

		size_type area(void) const noexcept
		{
			return (height * width);
		}

		size_type row_size(void) const noexcept
		{
			return stride;
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

		vector_type operator[](size_type i) noexcept
		{
			return vector_type(width, channels, buffer + i * stride);
		}
		const_vector_type operator[](size_type i) const noexcept
		{
			return const_vector_type(width, channels, buffer + i * stride);
		}

		vector_type at(size_type i)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (i >= height)
				throw ::std::out_of_range(matrix_out_of_range);
			return vector_type(width, channels, buffer + i * stride);
		}
		const_vector_type at(size_type i) const
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (i >= height)
				throw ::std::out_of_range(matrix_out_of_range);
			return const_vector_type(width, channels, buffer + i * stride);
		}

		pointer data(void) noexcept
		{
			return static_cast<pointer>(buffer);
		}
		const_pointer data(void) const noexcept
		{
			return static_cast<const_pointer>(buffer);
		}
		pointer data(size_type row) noexcept
		{
			return static_cast<pointer>(buffer + row * stride);
		}
		const_pointer data(size_type row) const noexcept
		{
			return static_cast<const_pointer>(buffer + row * stride);
		}
		pointer data(size_type row, size_type column) noexcept
		{
			return static_cast<pointer>(buffer + row * stride + column * channels);
		}
		const_pointer data(size_type row, size_type column) const noexcept
		{
			return static_cast<const_pointer>(buffer + row * stride + column * channels);
		}
		pointer data(size_type row, size_type column, size_type dim) noexcept
		{
			return static_cast<pointer>(buffer + row * stride + column * channels + dim);
		}
		const_pointer data(size_type row, size_type column, size_type dim) const noexcept
		{
			return static_cast<const_pointer>(buffer + row * stride + column * channels + dim);
		}

		scalar_type scalar(iterator it) noexcept
		{
			return scalar_type(channels, it.operator->());
		}
		scalar_type scalar(reverse_iterator it) noexcept
		{
			return scalar_type(channels, it.operator->());
		}
		const_scalar_type scalar(const_iterator it) noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()));
		}
		const_scalar_type scalar(const_reverse_iterator it) noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()));
		}
		const_scalar_type scalar(const_iterator it) const noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()));
		}
		const_scalar_type scalar(const_reverse_iterator it) const noexcept
		{
			return const_scalar_type(channels, const_cast<pointer>(it.operator->()));
		}

		vector_type vector(iterator it) noexcept
		{
			return vector_type(width, channels, it.operator->());
		}
		vector_type vector(reverse_iterator it) noexcept
		{
			return vector_type(width, channels, it.operator->());
		}
		const_vector_type vector(const_iterator it) noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()));
		}
		const_vector_type vector(const_reverse_iterator it) noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()));
		}
		const_vector_type vector(const_iterator it) const noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()));
		}
		const_vector_type vector(const_reverse_iterator it) const noexcept
		{
			return const_vector_type(width, channels, const_cast<pointer>(it.operator->()));
		}

		// modifiers:

		void fill(const value_type& value)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			::std::fill_n(buffer, count, value);
		}

		void fill_n(size_type n, const value_type& value)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (n == 0 || n > count)
				throw ::std::invalid_argument(invalid_length);
			::std::fill_n(buffer, n, value);
		}

		template <class InputIterator>
		void fill(InputIterator first, InputIterator last)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (static_cast<size_type>(::std::distance(first, last)) != count)
				throw ::std::invalid_argument(invalid_iterator_distance);
			::std::copy(first, last, buffer);
		}

		void fill(::std::initializer_list<T> il)
		{
			fill(il.begin(), il.end());
		}

		void fill(const matrix<T, Allocator>& x)
		{
			if (empty() || x.empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (count != x.size())
				throw ::std::invalid_argument(invalid_length);
			::std::uninitialized_copy(x.buffer, x.buffer + count, buffer);
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

		void linear_fill(const value_type& init, const value_type& row_delta, const value_type& col_delta)
		{
			value_type row_value(init);
			pointer current = buffer;
			for (size_type j = 0; j < height; ++j)
			{
				value_type col_value(row_value);
				for (size_type i = 0; i < stride; ++i)
				{
					*current++ = col_value;
					col_value += col_delta;
				}
				row_value += row_delta;
			}
		}

		void value(const scalar_type& element)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (element.size() != channels)
				throw ::std::invalid_argument(invalid_dimension);
			pointer current = buffer;
			size_type number = area();
			const_pointer first = element.data();
			const_pointer last = first + element.size();
			for (size_type i = 0; i < number; ++i)
			{
				::std::copy(first, last, current);
				current += channels;
			}
		}

		void linear_value(const scalar_type& init, const value_type& row_delta, const value_type& col_delta)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (init.size() != channels)
				throw ::std::invalid_argument(invalid_dimension);
			vector_type first_row = vector_type(width, channels, buffer);
			first_row.linear_value(init, col_delta);
			pointer current = buffer;
			for (size_type j = 1; j < height; ++j)
			{
				pointer next = current + stride;
				for (size_type i = 0; i < stride; ++i)
					next[i] = current[i] + row_delta;
				current = next;
			}
		}

		template<class Generator>
		void generate(Generator g)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			for (size_type i = 0; i < count; ++i)
				buffer[i] = g();
		}

		void reshape(size_type rows, size_type columns, size_type dimension)
		{
			if (empty())
				throw ::std::domain_error(matrix_not_initialized);
			if (rows * columns * dimension != count)
				throw ::std::invalid_argument(invalid_matrix_size);
			channels = dimension;
			width = columns;
			height = rows;
			stride = width * channels;
		}

		void swap(matrix<T, Allocator>& rhs) noexcept
		{
			if (this != &rhs)
			{
				::std::swap(owner, rhs.owner);
				::std::swap(channels, rhs.channels);
				::std::swap(width, rhs.width);
				::std::swap(height, rhs.height);
				::std::swap(stride, rhs.stride);
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
			stride = 0;
			count = 0;
		}

		// allocator

		allocator_type get_allocator(void) const noexcept
		{
			return *static_cast<const allocator_type*>(this);
		}

		// operator:

		matrix<T, Allocator>& operator+=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] += value;
			return *this;
		}

		matrix<T, Allocator>& operator-=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] -= value;
			return *this;
		}

		matrix<T, Allocator>& operator*=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] *= value;
			return *this;
		}

		matrix<T, Allocator>& operator/=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] /= value;
			return *this;
		}

		matrix<T, Allocator>& operator&=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] &= value;
			return *this;
		}

		matrix<T, Allocator>& operator^=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] ^= value;
			return *this;
		}

		matrix<T, Allocator>& operator|=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] |= value;
			return *this;
		}

		template <class A>
		matrix<T, Allocator>& operator+=(const matrix<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] += ptr[i];
			return *this;
		}

		template <class A>
		matrix<T, Allocator>& operator-=(const matrix<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] -= ptr[i];
			return *this;
		}

		template <class A>
		matrix<T, Allocator>& operator*=(const matrix<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] *= ptr[i];
			return *this;
		}

		template <class A>
		matrix<T, Allocator>& operator/=(const matrix<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] /= ptr[i];
			return *this;
		}

		template <class A>
		matrix<T, Allocator>& operator&=(const matrix<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] &= ptr[i];
			return *this;
		}

		template <class A>
		matrix<T, Allocator>& operator^=(const matrix<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] ^= ptr[i];
			return *this;
		}

		template <class A>
		matrix<T, Allocator>& operator|=(const matrix<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] |= ptr[i];
			return *this;
		}
	public:
		// operator:

		template <class A>
		friend matrix<T, Allocator> operator+(const matrix<T, Allocator>& src1, const matrix<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(matrix_different_size);
			matrix<T, Allocator> dst(src1.rows(), src1.columns(), src1.dimension());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] + ptr2[i];
			return dst;
		}

		template <class A>
		friend matrix<T, Allocator> operator-(const matrix<T, Allocator>& src1, const matrix<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(matrix_different_size);
			matrix<T, Allocator> dst(src1.rows(), src1.columns(), src1.dimension());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] - ptr2[i];
			return dst;
		}

		template <class A>
		friend matrix<T, Allocator> operator*(const matrix<T, Allocator>& src1, const matrix<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(matrix_different_size);
			matrix<T, Allocator> dst(src1.rows(), src1.columns(), src1.dimension());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] * ptr2[i];
			return dst;
		}

		template <class A>
		friend matrix<T, Allocator> operator/(const matrix<T, Allocator>& src1, const matrix<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(matrix_different_size);
			matrix<T, Allocator> dst(src1.rows(), src1.columns(), src1.dimension());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] / ptr2[i];
			return dst;
		}

		template <class A>
		friend matrix<T, Allocator> operator&(const matrix<T, Allocator>& src1, const matrix<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(matrix_different_size);
			matrix<T, Allocator> dst(src1.rows(), src1.columns(), src1.dimension());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] & ptr2[i];
			return dst;
		}

		template <class A>
		friend matrix<T, Allocator> operator^(const matrix<T, Allocator>& src1, const matrix<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(matrix_different_size);
			matrix<T, Allocator> dst(src1.rows(), src1.columns(), src1.dimension());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] ^ ptr2[i];
			return dst;
		}

		template <class A>
		friend matrix<T, Allocator> operator|(const matrix<T, Allocator>& src1, const matrix<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(matrix_different_size);
			matrix<T, Allocator> dst(src1.rows(), src1.columns(), src1.dimension());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] | ptr2[i];
			return dst;
		}

		template <class A>
		friend bool operator<(const matrix<T, Allocator>& lhs, const matrix<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
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
		friend bool operator>(const matrix<T, Allocator>& lhs, const matrix<T, A>& rhs)
		{
			return (rhs < lhs);
		}
		template <class A>
		friend bool operator<=(const matrix<T, Allocator>& lhs, const matrix<T, A>& rhs)
		{
			return !(lhs > rhs);
		}
		template <class A>
		friend bool operator>=(const matrix<T, Allocator>& lhs, const matrix<T, A>& rhs)
		{
			return !(lhs < rhs);
		}

		template <class A>
		friend bool operator==(const matrix<T, Allocator>& lhs, const matrix<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(matrix_different_size);
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
		friend bool operator!=(const matrix<T, Allocator>& lhs, const matrix<T, A>& rhs)
		{
			return !(lhs == rhs);
		}
	private:
		void assign_rv(matrix<T, Allocator>&& right, ::std::true_type)
		{
			swap(right);
		}
		void assign_rv(matrix<T, Allocator>&& right, ::std::false_type)
		{
			if (get_allocator() == right.get_allocator())
				assign_rv(::std::forward<matrix<T, Allocator> >(right), ::std::true_type());
			else
				assign(right);
		}
	private:
		bool       owner;
		size_type  channels;
		size_type  width;
		size_type  height;
		size_type  stride;
		size_type  count;
		pointer    buffer;
	};

} // namespace core

#endif
