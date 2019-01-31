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

#ifndef __CORE_SB_TREE_H__
#define __CORE_SB_TREE_H__

#include <memory>
#include <iterator>
#include <functional>
#include <utility>

namespace core
{
	typedef signed char sb_tree_node_state;
	static constexpr sb_tree_node_state sb_tree_state_parent  = -0x10;
	static constexpr sb_tree_node_state sb_tree_state_sibling =  0x01;
	static constexpr sb_tree_node_state sb_tree_state_left    =  0x12;
	static constexpr sb_tree_node_state sb_tree_state_right   =  0x13;
	static constexpr sb_tree_node_state sb_tree_state_root    =  0x04;

	// Class template sb_tree_node
	template <class T>
	struct sb_tree_node
	{
		typedef rb_tree_node<T>  node_type;
		typedef node_type*       node_pointer;
		typedef const node_type* const_node_pointer;
		typedef node_type&       node_reference;
		typedef const node_type& const_node_reference;

		node_pointer             parent;
		node_pointer             left;
		node_pointer             right;
		size_t                   size;
		T                        data;
	};

	// Class template sb_tree_type_traits

	template <class Tree, bool IsConst>
	struct sb_tree_type_traits
	{
		typedef typename Tree::node_type       node_type;
		typedef typename Tree::node_pointer    node_pointer;
		typedef typename Tree::value_type      value_type;
		typedef typename Tree::pointer         pointer;
		typedef typename Tree::reference       reference;
		typedef typename Tree::size_type       size_type;
		typedef typename Tree::difference_type difference_type;
	};

	template <class Tree>
	struct sb_tree_type_traits<Tree, true>
	{
		typedef typename Tree::node_type       node_type;
		typedef typename Tree::node_pointer    node_pointer;
		typedef typename Tree::value_type      value_type;
		typedef typename Tree::const_pointer   pointer;
		typedef typename Tree::const_reference reference;
		typedef typename Tree::size_type       size_type;
		typedef typename Tree::difference_type difference_type;
	};

	// Class template sb_tree_iterator
	template <class Tree, bool IsConst>
	class sb_tree_iterator
	{
	public:
		// types:

		typedef sb_tree_iterator<Tree, IsConst>                              iterator_type;
		typedef ::std::bidirectional_iterator_tag                            iterator_category;

		typedef typename sb_tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename sb_tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename sb_tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename sb_tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename sb_tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename sb_tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename sb_tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		sb_tree_iterator(void) noexcept
			: node(nullptr)
		{}
		explicit sb_tree_iterator(const node_pointer ptr) noexcept
			: node(ptr)
		{}
		sb_tree_iterator(const sb_tree_iterator<Tree, IsConst>& x) noexcept
			: node(x.get_pointer())
		{}

		sb_tree_iterator<Tree, IsConst>& operator=(const sb_tree_iterator<Tree, IsConst>& x) noexcept
		{
			if (this != &x)
				node = x.get_pointer();
			return (*this);
		}

		operator sb_tree_iterator<Tree, true>(void) const noexcept
		{
			return sb_tree_iterator<Tree, true>(node);
		}

		// sb_tree_iterator operations:

		node_pointer get_parent(void) noexcept
		{
			return node->parent;
		}
		const node_pointer get_parent(void) const noexcept
		{
			return node->parent;
		}

		node_pointer get_pointer(void) noexcept
		{
			return node;
		}
		const node_pointer get_pointer(void) const noexcept
		{
			return node;
		}

		size_t get_size(void) const noexcept
		{
			return node->size;
		}

		reference operator*(void) const noexcept
		{
			return node->data;
		}
		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		sb_tree_iterator<Tree, IsConst>& operator++(void) noexcept
		{
			if (node->right != nullptr)
			{
				node = node->right;
				while (node->left != nullptr)
					node = node->left;
			}
			else
			{
				node_pointer ptr = node->parent;
				while (node == ptr->right)
				{
					node = ptr;
					ptr = ptr->parent;
				}
				if (node->right != ptr)
					node = ptr;
			}
			return *this;
		}
		sb_tree_iterator<Tree, IsConst>& operator--(void) noexcept
		{
			if (node->parent->parent == node)
			{
				node = node->right;
			}
			else if (node->left != nullptr)
			{
				node_pointer ptr = node->left;
				while (ptr->right != nullptr)
					ptr = ptr->right;
				node = ptr;
			}
			else
			{
				node_pointer ptr = node->parent;
				while (node == ptr->left)
				{
					node = ptr;
					ptr = ptr->parent;
				}
				node = ptr;
			}
			return *this;
		}
		sb_tree_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		sb_tree_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const sb_tree_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const sb_tree_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer node;
	};

	// Class template sb_tree_primitive_iterator
	template <class Tree, bool IsConst>
	class sb_tree_primitive_iterator
	{
	public:
		// types:

		typedef sb_tree_primitive_iterator<Tree, IsConst>                    iterator_type;
		typedef ::std::bidirectional_iterator_tag                            iterator_category;

		typedef typename sb_tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename sb_tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename sb_tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename sb_tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename sb_tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename sb_tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename sb_tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		sb_tree_primitive_iterator(void) noexcept
			: node(nullptr)
			, state(sb_tree_state_root)
		{}
		explicit sb_tree_primitive_iterator(const node_pointer ptr) noexcept
			: node(ptr)
			, state(sb_tree_state_root)
		{}
		sb_tree_primitive_iterator(const sb_tree_primitive_iterator<Tree, IsConst>& x) noexcept
			: node(x.get_pointer())
			, state(x.get_state())
		{}

		sb_tree_primitive_iterator<Tree, IsConst>& operator=(const sb_tree_primitive_iterator<Tree, IsConst>& x) noexcept
		{
			if (this != &x)
			{
				node = x.get_pointer();
				state = x.get_state();
			}
			return (*this);
		}

		operator sb_tree_primitive_iterator<Tree, true>(void) const noexcept
		{
			return sb_tree_primitive_iterator<Tree, true>(node);
		}

		// sb_tree_primitive_iterator operations:

		node_pointer get_parent(void) noexcept
		{
			return node->parent;
		}
		const node_pointer get_parent(void) const noexcept
		{
			return node->parent;
		}

		node_pointer get_pointer(void) noexcept
		{
			return node;
		}
		const node_pointer get_pointer(void) const noexcept
		{
			return node;
		}

		sb_tree_node_state get_state(void) const noexcept
		{
			return state;
		}

		intptr_t get_depth(void) const noexcept
		{
			return static_cast<intptr_t>(state >> 4);
		}

		reference operator*(void) const noexcept
		{
			return node->data;
		}
		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		sb_tree_primitive_iterator<Tree, IsConst>& operator++(void) noexcept
		{
			if (state != sb_tree_state_parent)
			{
				if (node->left != nullptr)
				{
					node = node->left;
					state = sb_tree_state_left;
				}
				else if (node->right != nullptr)
				{
					node = node->right;
					state = sb_tree_state_right;
				}
				else if (node != node->parent->parent && node != node->parent->right)
				{
					node = node->parent->right;
					state = sb_tree_state_sibling;
				}
				else
				{
					node = node->parent;
					state = sb_tree_state_parent;
				}
			}
			else
			{
				if (node != node->parent->parent && node != node->parent->right)
				{
					node = node->parent->right;
					state = sb_tree_state_sibling;
				}
				else
					node = node->parent;
			}
			return *this;
		}
		sb_tree_primitive_iterator<Tree, IsConst>& operator--(void) noexcept
		{
			if (state != sb_tree_state_parent)
			{
				if (node->right != nullptr)
				{
					node = node->right;
					state = sb_tree_state_right;
				}
				else if (node->left != nullptr)
				{
					node = node->left;
					state = sb_tree_state_left;
				}
				else if (node != node->parent->parent && node != node->parent->left)
				{
					node = node->parent->left;
					state = sb_tree_state_sibling;
				}
				else
				{
					node = node->parent;
					state = sb_tree_state_parent;
				}
			}
			else
			{
				if (node != node->parent->parent && node != node->parent->left)
				{
					node = node->parent->left;
					state = sb_tree_state_sibling;
				}
				else
					node = node->parent;
			}
			return *this;
		}
		sb_tree_primitive_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		sb_tree_primitive_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const sb_tree_primitive_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const sb_tree_primitive_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer       node;
		sb_tree_node_state state;
	};

	// Class template sb_tree_node_allocator
	template <class T, class Allocator>
	class sb_tree_node_allocator : public Allocator
	{
	public:
		// types:

		typedef Allocator                                                  allocator_type;
		typedef typename sb_tree_node<T>::node_type                        tree_node_type;
		typedef typename Allocator::template rebind<tree_node_type>::other node_allocator_type;
		typedef ::std::allocator_traits<allocator_type>                    allocator_traits_type;
		typedef ::std::allocator_traits<node_allocator_type>               node_allocator_traits_type;

		typedef typename node_allocator_traits_type::size_type             node_size_type;
		typedef typename node_allocator_traits_type::difference_type       node_difference_type;
		typedef typename node_allocator_traits_type::value_type            node_type;
		typedef typename node_allocator_traits_type::pointer               node_pointer;

		// construct/copy/destroy:

		sb_tree_node_allocator(void)
			: Allocator()
		{}
		explicit sb_tree_node_allocator(const Allocator& alloc)
			: Allocator(alloc)
		{}
		explicit sb_tree_node_allocator(Allocator&& alloc)
			: Allocator(::std::forward<Allocator>(alloc))
		{}
		~sb_tree_node_allocator(void)
		{}

		// sb_tree_node_allocator operations:

		allocator_type get_allocator(void) const noexcept
		{
			return *static_cast<const allocator_type*>(this);
		}
		node_size_type max_size(void) const noexcept
		{
			return node_alloc.max_size();
		}
	protected:
		template<class ...Args>
		node_pointer create_node(Args&&... args)
		{
			node_pointer p = node_alloc.allocate(1);
			get_allocator().construct(::std::addressof(p->data), ::std::forward<Args>(args)...);
			return p;
		}
		void destroy_node(const node_pointer p)
		{
			get_allocator().destroy(::std::addressof(p->data));
			node_alloc.deallocate(p, 1);
		}
	private:
		node_allocator_type node_alloc;
	};

	// Class template sb_tree
	template <class Key, class Value, class KeyOfValue, class KeyCompare = ::std::less<Key>, class Allocator = allocator<Value> >
	class sb_tree : public sb_tree_node_allocator<Value, Allocator>
	{
	public:
		// types:

		typedef Key                                                    key_type;
		typedef KeyCompare                                             key_compare;
		typedef Allocator                                              allocator_type;
		typedef sb_tree_node_allocator<Value, Allocator>               node_allocator_type;
		typedef sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator> tree_type;
		typedef typename node_allocator_type::node_type                node_type;
		typedef typename node_allocator_type::node_pointer             node_pointer;
		typedef typename allocator_type::value_type                    value_type;
		typedef typename allocator_type::pointer                       pointer;
		typedef typename allocator_type::const_pointer                 const_pointer;
		typedef typename allocator_type::reference                     reference;
		typedef typename allocator_type::const_reference               const_reference;
		typedef typename allocator_type::size_type                     size_type;
		typedef typename allocator_type::difference_type               difference_type;

		typedef sb_tree_iterator<tree_type, false>                     iterator;
		typedef sb_tree_iterator<tree_type, true>                      const_iterator;
		typedef sb_tree_primitive_iterator<tree_type, false>           primitive_iterator;
		typedef sb_tree_primitive_iterator<tree_type, true>            const_primitive_iterator;
		typedef ::std::reverse_iterator<iterator>                      reverse_iterator;
		typedef ::std::reverse_iterator<const_iterator>                const_reverse_iterator;
		typedef ::std::reverse_iterator<primitive_iterator>            reverse_primitive_iterator;
		typedef ::std::reverse_iterator<const_primitive_iterator>      const_reverse_primitive_iterator;

		// construct/copy/destroy:

		explicit sb_tree(const KeyCompare& comp = KeyCompare(), const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
			, compare(comp)
			, count(0)
		{
			create_header();
		}
		explicit sb_tree(const Allocator& alloc)
			: node_allocator_type(alloc)
			, compare(KeyCompare())
			, count(0)
		{
			create_header();
		}
		sb_tree(const sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>& x)
			: node_allocator_type(x.get_allocator())
			, compare(x.compare)
			, count(0)
		{
			create_header();
			if (x.header->parent != nullptr)
				copy_root(x.header->parent);
			count = x.count;
		}
		sb_tree(const sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>& x, const Allocator& alloc)
			: node_allocator_type(alloc)
			, compare(x.compare)
			, count(0)
		{
			create_header();
			if (x.header->parent != nullptr)
				copy_root(x.header->parent);
			count = x.count;
		}
		sb_tree(sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>&& x)
			: node_allocator_type(x.get_allocator())
		{
			create_header();
			if (this != &x)
			{
				::std::swap(header, x.header);
				::std::swap(compare, x.compare);
				::std::swap(count, x.count);
			}
		}
		sb_tree(sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>&& x, const Allocator& alloc)
			: node_allocator_type(alloc)
		{
			create_header();
			if (this != &x)
			{
				::std::swap(header, x.header);
				::std::swap(compare, x.compare);
				::std::swap(count, x.count);
			}
		}
		~sb_tree(void)
		{
			clear();
			destroy_header();
		}
		sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>& operator=(const sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>& x)
		{
			if (this != &x)
			{
				clear();
				if (x.header->parent != nullptr)
					copy_root(x.header->parent);
				compare = x.compare;
				count = x.count;
			}
			return (*this);
		}
		sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>& operator=(sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>&& x)
		{
			if (this != &x)
			{
				::std::swap(header, x.header);
				::std::swap(compare, x.compare);
				::std::swap(count, x.count);
			}
			return (*this);
		}
		void assign_equal(const_iterator first, const_iterator last)
		{
			clear();
			insert_equal(first, last);
		}
		void assign_equal(size_type n, const Value& value)
		{
			clear();
			insert_equal(n, value);
		}
		void assign_equal(::std::initializer_list<Value> init)
		{
			clear();
			insert_equal(init.begin(), init.end());
		}
		void assign_unique(const_iterator first, const_iterator last)
		{
			clear();
			insert_unique(first, last);
		}
		void assign_unique(const Value& value)
		{
			clear();
			insert_unique(value);
		}
		void assign_unique(::std::initializer_list<Value> init)
		{
			clear();
			insert_unique(init.begin(), init.end());
		}

		// iterators:

		iterator begin(void) noexcept
		{
			return iterator(header->left);
		}
		const_iterator begin(void) const noexcept
		{
			return const_iterator(header->left);
		}
		const_iterator cbegin(void) const noexcept
		{
			return const_iterator(header->left);
		}
		iterator end(void) noexcept
		{
			return iterator(header);
		}
		const_iterator end(void) const noexcept
		{
			return const_iterator(header);
		}
		const_iterator cend(void) const noexcept
		{
			return const_iterator(header);
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

		primitive_iterator pbegin(void) noexcept
		{
			return primitive_iterator(get_root());
		}
		const_primitive_iterator pbegin(void) const noexcept
		{
			return const_primitive_iterator(get_root());
		}
		const_primitive_iterator cpbegin(void) const noexcept
		{
			return const_primitive_iterator(get_root());
		}
		primitive_iterator pend(void) noexcept
		{
			return primitive_iterator(header);
		}
		const_primitive_iterator pend(void) const noexcept
		{
			return const_primitive_iterator(header);
		}
		const_primitive_iterator cpend(void) const noexcept
		{
			return const_primitive_iterator(header);
		}
		reverse_primitive_iterator rpbegin(void) noexcept
		{
			return reverse_primitive_iterator(pend());
		}
		const_reverse_primitive_iterator rpbegin(void) const noexcept
		{
			return const_reverse_primitive_iterator(pend());
		}
		const_reverse_primitive_iterator crpbegin(void) const noexcept
		{
			return const_reverse_primitive_iterator(cpend());
		}
		reverse_primitive_iterator rpend(void) noexcept
		{
			return reverse_primitive_iterator(pbegin());
		}
		const_reverse_primitive_iterator rpend(void) const noexcept
		{
			return const_reverse_primitive_iterator(pbegin());
		}
		const_reverse_primitive_iterator crpend(void) const noexcept
		{
			return const_reverse_primitive_iterator(cpbegin());
		}

		// capacity:

		bool empty(void) const noexcept
		{
			return (count == 0);
		}

		size_type size(void) const noexcept
		{
			return count;
		}

		// observers:

		key_compare key_comp(void) const
		{
			return compare;
		}

		// modifiers:

		template <class... Args>
		iterator emplace_equal(Args&&... args)
		{
			return iterator(insert_equal_node(::std::forward<Args>(args)...));
		}

		iterator insert_equal(const value_type& value)
		{
			return iterator(insert_equal_node(value));
		}

		iterator insert_equal(value_type&& value)
		{
			return iterator(insert_equal_node(::std::forward<value_type>(value)));
		}

		void insert_equal(size_type n, const value_type& value)
		{
			for (; n > 0; --n)
				insert_equal_node(value);
		}

		template <class InputIterator>
		void insert_equal(InputIterator first, InputIterator last)
		{
			for (; first != last; ++first)
				insert_equal_node(*first);
		}

		template <class InputIterator>
		void insert_equal(::std::initializer_list<value_type> init)
		{
			insert_equal(init.begin(), init.end());
		}

		template <class... Args>
		iterator emplace_unique(Args&&... args)
		{
			return iterator(insert_unique_node(::std::forward<Args>(args)...));
		}

		iterator insert_unique(const value_type& value)
		{
			return iterator(insert_unique_node(value));
		}

		iterator insert_unique(value_type&& value)
		{
			return iterator(insert_unique_node(::std::forward<value_type>(value)));
		}

		template <class InputIterator>
		void insert_unique(InputIterator first, InputIterator last)
		{
			for (; first != last; ++first)
				insert_unique_node(*first);
		}

		void insert_unique(::std::initializer_list<value_type> init)
		{
			insert_unique(init.begin(), init.end());
		}

		void erase(iterator position)
		{
			erase_node(position.get_pointer(), header->parent);
		}

		void erase(iterator first, iterator last)
		{
			if (first == begin() && last == end())
				clear();
			else
				while (first != last)
					erase(first++);
		}

		size_type erase(const key_type& key)
		{
			size_type n = 0;
			iterator first = lower_bound(key);
			iterator last = upper_bound(key);
			while (first != last)
			{
				++n;
				erase(first++);
			}
			return n;
		}

		void swap(sb_tree<Key, Value, KeyOfValue, KeyCompare, Allocator>& rhs) noexcept
		{
			if (this != &rhs)
			{
				::std::swap(header, rhs.header);
				::std::swap(compare, rhs.compare);
				::std::swap(count, rhs.count);
			}
		}

		void clear(void)
		{
			if (header->parent != nullptr)
			{
				erase_root();
				header->parent = nullptr;
				header->left = header;
				header->right = header;
				count = 0;
			}
		}

		// operations:

		iterator find(const key_type& key) noexcept
		{
			return iterator(find_node(key));
		}
		const_iterator find(const key_type& key) const noexcept
		{
			return const_iterator(find_node(key));
		}

		iterator lower_bound(const key_type& key) noexcept
		{
			return iterator(lower_bound_node(key));
		}
		const_iterator lower_bound(const key_type& key) const noexcept
		{
			return const_iterator(lower_bound_node(key));
		}

		iterator upper_bound(const key_type& key) noexcept
		{
			return iterator(upper_bound_node(key));
		}
		const_iterator upper_bound(const key_type& key) const noexcept
		{
			return const_iterator(upper_bound_node(key));
		}
	private:
		node_pointer get_root(void) const noexcept
		{
			return (header->parent != nullptr ? header->parent : header);
		}

		node_pointer get_leftmost(const node_pointer parent) const noexcept
		{
			node_pointer leftmost = parent;
			while (leftmost->left)
				leftmost = leftmost->left;
			return leftmost;
		}

		node_pointer get_rightmost(const node_pointer parent) const noexcept
		{
			node_pointer rightmost = parent;
			while (rightmost->right)
				rightmost = rightmost->right;
			return rightmost;
		}

		void create_header(void)
		{
			header = this->create_node(value_type());
			header->parent = nullptr;
			header->left = header;
			header->right = header;
		}

		void destroy_header(void)
		{
			this->destroy_node(header);
		}

		node_pointer find_node(const key_type& key) const noexcept
		{
			node_pointer pre = header;
			node_pointer cur = header->parent;
			while (cur != nullptr)
			{
				if (!compare(KeyOfValue()(cur->data), key))
				{
					pre = cur;
					cur = cur->left;
				}
				else
					cur = cur->right;
			}
			if (pre == header || compare(key, KeyOfValue()(pre->data)))
			{
				pre = header;
			}
			return pre;
		}

		node_pointer lower_bound_node(const key_type& key) const noexcept
		{
			node_pointer pre = header;
			node_pointer cur = header->parent;
			while (cur != nullptr)
			{
				if (!compare(KeyOfValue()(cur->data), key))
				{
					pre = cur;
					cur = cur->left;
				}
				else
					cur = cur->right;
			}
			return pre;
		}

		node_pointer upper_bound_node(const key_type& key) const noexcept
		{
			node_pointer pre = header;
			node_pointer cur = header->parent;
			while (cur != nullptr)
			{
				if (compare(key, KeyOfValue()(cur->data)))
				{
					pre = cur;
					cur = cur->left;
				}
				else
					cur = cur->right;
			}
			return pre;
		}

		template<class ...Args>
		node_pointer insert_equal_node(Args&&... args)
		{
			node_pointer node = this->create_node(::std::forward<Args>(args)...);
			node->left = nullptr;
			node->right = nullptr;
			if (header->parent == nullptr)
			{
				node->parent = header;
				header->parent = node;
				header->left = node;
				header->right = node;
			}
			else
			{
				node_pointer ptr = header->parent;
				while (ptr != nullptr)
				{
					if (compare(KeyOfValue()(node->data), KeyOfValue()(ptr->data)))
					{
						if (ptr->left != nullptr)
							ptr = ptr->left;
						else
						{
							node->parent = ptr;
							ptr->left = node;
							if (ptr == header->left)
								header->left = node;
							ptr = nullptr;
						}
					}
					else
					{
						if (ptr->right != nullptr)
							ptr = ptr->right;
						else
						{
							node->parent = ptr;
							ptr->right = node;
							if (ptr == header->right)
								header->right = node;
							ptr = nullptr;
						}
					}
				}
				insert_rebalance(node, header->parent);
			}
			++count;
			return node;
		}

		template<class ...Args>
		node_pointer insert_unique_node(Args&&... args)
		{
			node_pointer node = nullptr;
			value_type value = value_type(::std::forward<Args>(args)...);
			if (header->parent == nullptr)
			{
				node = this->create_node(::std::forward<value_type>(value));
				node->parent = header;
				node->left = nullptr;
				node->right = nullptr;
				header->parent = node;
				header->left = node;
				header->right = node;
			}
			else
			{
				node_pointer ptr = header->parent;
				while (ptr != nullptr)
				{
					if (compare(KeyOfValue()(value), KeyOfValue()(ptr->data)))
					{
						if (ptr->left != nullptr)
							ptr = ptr->left;
						else
						{
							node = this->create_node(::std::forward<value_type>(value));
							node->parent = ptr;
							node->left = nullptr;
							node->right = nullptr;
							ptr->left = node;
							if (ptr == header->left)
								header->left = node;
							ptr = nullptr;
						}
					}
					else
					{
						if (!compare(KeyOfValue()(ptr->data), KeyOfValue()(value)))
							return ptr;
						if (ptr->right != nullptr)
							ptr = ptr->right;
						else
						{
							node = this->create_node(::std::forward<value_type>(value));
							node->parent = ptr;
							node->left = nullptr;
							node->right = nullptr;
							ptr->right = node;
							if (ptr == header->right)
								header->right = node;
							ptr = nullptr;
						}
					}
				}
				insert_rebalance(node, header->parent);
			}
			++count;
			return node;
		}

		void erase_node(node_pointer position, node_pointer& root)
		{
			node_pointer p = nullptr;
			node_pointer x = nullptr;
			node_pointer y = position;
			node_pointer z = position;
			if (y->left == nullptr || y->right == nullptr)
			{
				p = y->parent;
				x = (y->left == nullptr) ? y->right : y->left;
				if (x != nullptr)
					x->parent = y->parent;
				if (z == root)
					root = x;
				else if (z == z->parent->left)
					z->parent->left = x;
				else
					z->parent->right = x;
				if (z == header->left)
					header->left = (z->right == nullptr) ? z->parent : get_leftmost(x);
				if (z == header->right)
					header->right = (z->left == nullptr) ? z->parent : get_rightmost(x);
			}
			else
			{
				y = z->right;
				while (y->left != nullptr)
					y = y->left;
				x = y->right;
				if (y == z->right)
				{
					p = y;
					z->left->parent = y;
					y->left = z->left;
				}
				else
				{
					p = y->parent;
					z->left->parent = y;
					z->right->parent = y;
					y->left = z->left;
					y->right = z->right;
					y->parent->left = x;
					if (x != nullptr)
						x->parent = y->parent;
				}
				if (z == root)
					root = y;
				else if (z == z->parent->left)
					z->parent->left = y;
				else
					z->parent->right = y;
				y->parent = z->parent;
				color = y->color;
				y->color = z->color;
				z->color = color;
				y = z;
			}
			this->destroy_node(y);
			--count;
		}

		void copy_root(const node_pointer root)
		{
			node_pointer src = root;
			node_pointer dst = header;
			sb_tree_node_state state = sb_tree_state_root;
			node_pointer node = this->create_node(root->data);
			node->color = root->color;
			node->parent = dst;
			node->left = nullptr;
			node->right = nullptr;
			dst->parent = node;
			dst = node;
			do
			{
				if (state != sb_tree_state_parent)
				{
					if (src->left != nullptr)
					{
						src = src->left;
						state = sb_tree_state_left;
						node = this->create_node(src->data);
						node->color = src->color;
						node->parent = dst;
						node->left = nullptr;
						node->right = nullptr;
						dst->left = node;
						dst = node;
					}
					else if (src->right != nullptr)
					{
						src = src->right;
						state = sb_tree_state_right;
						node = this->create_node(src->data);
						node->color = src->color;
						node->parent = dst;
						node->left = nullptr;
						node->right = nullptr;
						dst->right = node;
						dst = node;
					}
					else if (src != src->parent->parent && src != src->parent->right)
					{
						src = src->parent->right;
						state = sb_tree_state_sibling;
						node = this->create_node(src->data);
						node->color = src->color;
						node->parent = dst->parent;
						node->left = nullptr;
						node->right = nullptr;
						dst->parent->right = node;
						dst = node;
					}
					else
					{
						src = src->parent;
						dst = dst->parent;
						state = sb_tree_state_parent;
					}
				}
				else
				{
					if (src != src->parent->parent && src != src->parent->right)
					{
						src = src->parent->right;
						state = sb_tree_state_sibling;
						node = this->create_node(src->data);
						node->color = src->color;
						node->parent = dst->parent;
						node->left = nullptr;
						node->right = nullptr;
						dst->parent->right = node;
						dst = node;
					}
					else
					{
						src = src->parent;
						dst = dst->parent;
					}
				}
			} while (src != root);
			header->left = get_leftmost(header->parent);
			header->right = get_rightmost(header->parent);
		}

		void erase_root(void)
		{
			node_pointer next = nullptr;
			node_pointer cur = header->parent;
			do
			{
				while (cur->left != nullptr)
					cur = cur->left;
				if (cur->right != nullptr)
					cur = cur->right;
				else
				{
					next = cur->parent;
					if (cur == next->left)
						next->left = nullptr;
					else
						next->right = nullptr;
					this->destroy_node(cur);
					cur = next;
				}
			} while (cur != header);
		}

		void rotate_left(node_pointer x, node_pointer& root) const noexcept
		{
			node_pointer y = x->right;
			x->right = y->left;
			if (y->left != nullptr)
				y->left->parent = x;
			y->parent = x->parent;
			if (x == root)
				root = y;
			else if (x == x->parent->left)
				x->parent->left = y;
			else
				x->parent->right = y;
			y->left = x;
			x->parent = y;
			y->size = x->size;
			x->size = x->left->size + x->right->size + 1;
		}

		void rotate_right(node_pointer x, node_pointer& root) const noexcept
		{
			node_pointer y = x->left;
			x->left = y->right;
			if (y->right != nullptr)
				y->right->parent = x;
			y->parent = x->parent;
			if (x == root)
				root = y;
			else if (x == x->parent->right)
				x->parent->right = y;
			else
				x->parent->left = y;
			y->right = x;
			x->parent = y;
			y->size = x->size;
			x->size = x->left->size + x->right->size + 1;
		}

		void matain(node_pointer x, node_pointer& root) const noexcept
		{
			if (x->left->left->size > x->right->size)
			{
				rotate_right(x, root);
				matain(x->right, root);
				matain(x);
			}
			else if (x->left->right->size > x->right->size)
			{
				rotate_left(x->left, root);
				rotate_right(x, root);
				matain(x->left, root);
				matain(x->right, root);
				matain(x);
			}
			else if (x->right->right->size > x->left->size)
			{
				rotate_left(x, root);
				matain(x->left, root);
				matain(x);
			}
			else if (x->right->left->size > x->left->size)
			{
				rotate_right(x->right, root);
				rotate_left(x, root);
				matain(x->left, root);
				matain(x->right, root);
				matain(x);
			}
		}
	private:
		node_pointer header;
		key_compare  compare;
		size_type    count;
	};

} // namespace core

#endif
