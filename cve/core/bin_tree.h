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

#ifndef __CORE_BIN_TREE_H__
#define __CORE_BIN_TREE_H__

#include <memory>
#include <iterator>
#include <functional>
#include <utility>

namespace core
{
	typedef signed char bin_tree_node_state;
	static constexpr bin_tree_node_state bin_tree_state_parent  = -0x10;
	static constexpr bin_tree_node_state bin_tree_state_sibling =  0x01;
	static constexpr bin_tree_node_state bin_tree_state_left    =  0x12;
	static constexpr bin_tree_node_state bin_tree_state_right   =  0x13;
	static constexpr bin_tree_node_state bin_tree_state_root    =  0x04;

	// Class template bin_tree_node
	template <class T>
	struct bin_tree_node
	{
		typedef bin_tree_node<T> node_type;
		typedef node_type*       node_pointer;
		typedef const node_type* const_node_pointer;
		typedef node_type&       node_reference;
		typedef const node_type& const_node_reference;

		node_pointer             parent;
		node_pointer             left;
		node_pointer             right;
		T                        data;
	};

	// Class template bin_tree_type_traits

	template <class Tree, bool IsConst>
	struct bin_tree_type_traits
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
	struct bin_tree_type_traits<Tree, true>
	{
		typedef typename Tree::node_type       node_type;
		typedef typename Tree::node_pointer    node_pointer;
		typedef typename Tree::value_type      value_type;
		typedef typename Tree::const_pointer   pointer;
		typedef typename Tree::const_reference reference;
		typedef typename Tree::size_type       size_type;
		typedef typename Tree::difference_type difference_type;
	};

	// Class template bin_tree_iterator
	template <class Tree, bool IsConst>
	class bin_tree_iterator
	{
	public:
		// types:

		typedef bin_tree_iterator<Tree, IsConst>                              iterator_type;
		typedef ::std::bidirectional_iterator_tag                             iterator_category;

		typedef typename bin_tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename bin_tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename bin_tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename bin_tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename bin_tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename bin_tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename bin_tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		bin_tree_iterator(void) noexcept
			: node(nullptr)
		{}
		explicit bin_tree_iterator(const node_pointer ptr) noexcept
			: node(ptr)
		{}
		bin_tree_iterator(const bin_tree_iterator<Tree, IsConst>& x) noexcept
			: node(x.get_pointer())
		{}

		bin_tree_iterator<Tree, IsConst>& operator=(const bin_tree_iterator<Tree, IsConst>& x) noexcept
		{
			if (this != &x)
				node = x.get_pointer();
			return (*this);
		}

		operator bin_tree_iterator<Tree, true>(void) const noexcept
		{
			return bin_tree_iterator<Tree, true>(node);
		}

		// bin_tree_iterator operations:

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

		reference operator*(void) const noexcept
		{
			return node->data;
		}
		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		bin_tree_iterator<Tree, IsConst>& operator++(void) noexcept
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
		bin_tree_iterator<Tree, IsConst>& operator--(void) noexcept
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
		bin_tree_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		bin_tree_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const bin_tree_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const bin_tree_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer node;
	};

	// Class template bin_tree_primitive_iterator
	template <class Tree, bool IsConst>
	class bin_tree_primitive_iterator
	{
	public:
		// types:

		typedef bin_tree_primitive_iterator<Tree, IsConst>                    iterator_type;
		typedef ::std::bidirectional_iterator_tag                             iterator_category;

		typedef typename bin_tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename bin_tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename bin_tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename bin_tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename bin_tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename bin_tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename bin_tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		bin_tree_primitive_iterator(void) noexcept
			: node(nullptr)
			, state(bin_tree_state_root)
		{}
		explicit bin_tree_primitive_iterator(const node_pointer ptr) noexcept
			: node(ptr)
			, state(bin_tree_state_root)
		{}
		bin_tree_primitive_iterator(const bin_tree_primitive_iterator<Tree, IsConst>& x) noexcept
			: node(x.get_pointer())
			, state(x.get_state())
		{}

		bin_tree_primitive_iterator<Tree, IsConst>& operator=(const bin_tree_primitive_iterator<Tree, IsConst>& x) noexcept
		{
			if (this != &x)
			{
				node = x.get_pointer();
				state = x.get_state();
			}
			return (*this);
		}

		operator bin_tree_primitive_iterator<Tree, true>(void) const noexcept
		{
			return bin_tree_primitive_iterator<Tree, true>(node);
		}

		// bin_tree_primitive_iterator operations:

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

		bin_tree_node_state get_state(void) const noexcept
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

		bin_tree_primitive_iterator<Tree, IsConst>& operator++(void) noexcept
		{
			if (state != bin_tree_state_parent)
			{
				if (node->left != nullptr)
				{
					node = node->left;
					state = bin_tree_state_left;
				}
				else if (node->right != nullptr)
				{
					node = node->right;
					state = bin_tree_state_right;
				}
				else if (node != node->parent->parent && node != node->parent->right)
				{
					node = node->parent->right;
					state = bin_tree_state_sibling;
				}
				else
				{
					node = node->parent;
					state = bin_tree_state_parent;
				}
			}
			else
			{
				if (node != node->parent->parent && node != node->parent->right)
				{
					node = node->parent->right;
					state = bin_tree_state_sibling;
				}
				else
					node = node->parent;
			}
			return *this;
		}
		bin_tree_primitive_iterator<Tree, IsConst>& operator--(void) noexcept
		{
			if (state != bin_tree_state_parent)
			{
				if (node->right != nullptr)
				{
					node = node->right;
					state = bin_tree_state_right;
				}
				else if (node->left != nullptr)
				{
					node = node->left;
					state = bin_tree_state_left;
				}
				else if (node != node->parent->parent && node != node->parent->left)
				{
					node = node->parent->left;
					state = bin_tree_state_sibling;
				}
				else
				{
					node = node->parent;
					state = bin_tree_state_parent;
				}
			}
			else
			{
				if (node != node->parent->parent && node != node->parent->left)
				{
					node = node->parent->left;
					state = bin_tree_state_sibling;
				}
				else
					node = node->parent;
			}
			return *this;
		}
		bin_tree_primitive_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		bin_tree_primitive_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const bin_tree_primitive_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const bin_tree_primitive_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer       node;
		bin_tree_node_state state;
	};

	// Class template bin_tree_node_allocator
	template <class T, class Allocator>
	class bin_tree_node_allocator : public Allocator
	{
	public:
		// types:

		typedef Allocator                                                  allocator_type;
		typedef typename bin_tree_node<T>::node_type                       tree_node_type;
		typedef typename Allocator::template rebind<tree_node_type>::other node_allocator_type;
		typedef ::std::allocator_traits<allocator_type>                    allocator_traits_type;
		typedef ::std::allocator_traits<node_allocator_type>               node_allocator_traits_type;

		typedef typename node_allocator_traits_type::size_type             node_size_type;
		typedef typename node_allocator_traits_type::difference_type       node_difference_type;
		typedef typename node_allocator_traits_type::value_type            node_type;
		typedef typename node_allocator_traits_type::pointer               node_pointer;

		// construct/copy/destroy:

		bin_tree_node_allocator(void)
			: Allocator()
		{}
		explicit bin_tree_node_allocator(const Allocator& alloc)
			: Allocator(alloc)
		{}
		explicit bin_tree_node_allocator(Allocator&& alloc)
			: Allocator(::std::forward<Allocator>(alloc))
		{}
		~bin_tree_node_allocator(void)
		{}

		// bin_tree_node_allocator operations:

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

	// Class template bin_tree
	template <class T, class Allocator = allocator<T> >
	class bin_tree : public bin_tree_node_allocator<T, Allocator>
	{
	public:
		// types:

		typedef Allocator                                         allocator_type;
		typedef bin_tree_node_allocator<Value, Allocator>         node_allocator_type;
		typedef bin_tree<T, Allocator>                            tree_type;
		typedef typename node_allocator_type::node_type           node_type;
		typedef typename node_allocator_type::node_pointer        node_pointer;
		typedef typename allocator_type::value_type               value_type;
		typedef typename allocator_type::pointer                  pointer;
		typedef typename allocator_type::const_pointer            const_pointer;
		typedef typename allocator_type::reference                reference;
		typedef typename allocator_type::const_reference          const_reference;
		typedef typename allocator_type::size_type                size_type;
		typedef typename allocator_type::difference_type          difference_type;

		typedef bin_tree_iterator<tree_type, false>               iterator;
		typedef bin_tree_iterator<tree_type, true>                const_iterator;
		typedef bin_tree_primitive_iterator<tree_type, false>     primitive_iterator;
		typedef bin_tree_primitive_iterator<tree_type, true>      const_primitive_iterator;
		typedef ::std::reverse_iterator<iterator>                 reverse_iterator;
		typedef ::std::reverse_iterator<const_iterator>           const_reverse_iterator;
		typedef ::std::reverse_iterator<primitive_iterator>       reverse_primitive_iterator;
		typedef ::std::reverse_iterator<const_primitive_iterator> const_reverse_primitive_iterator;

		// construct/copy/destroy:

		explicit bin_tree(const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
			, count(0)
		{
			create_header();
		}
		bin_tree(const T& value, const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
			, count(0)
		{
			create_header();
			assign(value);
		}
		bin_tree(const bin_tree<T, Allocator>& x)
			: node_allocator_type(x.get_allocator())
			, count(0)
		{
			create_header();
			if (x.header->parent != nullptr)
				copy_root(x.header->parent);
			count = x.count;
		}
		bin_tree(const bin_tree<T, Allocator>& x, const Allocator& alloc)
			: node_allocator_type(alloc)
			, count(0)
		{
			create_header();
			if (x.header->parent != nullptr)
				copy_root(x.header->parent);
			count = x.count;
		}
		bin_tree(bin_tree<T, Allocator>&& x)
			: node_allocator_type(x.get_allocator())
		{
			create_header();
			if (this != &x)
			{
				::std::swap(header, x.header);
				::std::swap(count, x.count);
			}
		}
		bin_tree(bin_tree<T, Allocator>&& x, const Allocator& alloc)
			: node_allocator_type(alloc)
		{
			create_header();
			if (this != &x)
			{
				::std::swap(header, x.header);
				::std::swap(count, x.count);
			}
		}
		~bin_tree(void)
		{
			clear();
			destroy_header();
		}
		bin_tree<T, Allocator>& operator=(const bin_tree<T, Allocator>& x)
		{
			if (this != &x)
			{
				clear();
				if (x.header->parent != nullptr)
					copy_root(x.header->parent);
				count = x.count;
			}
			return (*this);
		}
		bin_tree<T, Allocator>& operator=(bin_tree<T, Allocator>&& x)
		{
			if (this != &x)
			{
				::std::swap(header, x.header);
				::std::swap(count, x.count);
			}
			return (*this);
		}
		void assign(const value_type& value)
		{
			clear();
			insert(value);
		}
		void assign(value_type&& value)
		{
			clear();
			insert(::std::forward<value_type>(value));
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

		// modifiers:

		template <class... Args>
		iterator emplace(Args&&... args)
		{
			return iterator(insert_root_node(::std::forward<Args>(args)...));
		}

		iterator insert(const value_type& value)
		{
			return iterator(insert_root_node(value));
		}

		iterator insert(value_type&& value)
		{
			return iterator(insert_root_node(::std::forward<value_type>(value)));
		}

		void erase(iterator position)
		{
			erase_node(position.get_pointer());
		}

		void swap(bin_tree<T, Allocator>& rhs) noexcept
		{
			if (this != &rhs)
			{
				::std::swap(header, rhs.header);
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

		template<class ...Args>
		node_pointer insert_root_node(Args&&... args)
		{
			node_pointer node = nullptr;
			if (header->parent == nullptr)
			{
				node = this->create_node(::std::forward<Args>(args)...);
				node->parent = header;
				node->left = nullptr;
				node->right = nullptr;
				header->parent = node;
				header->left = node;
				header->right = node;
				++count;
			}
			return node;
		}

		template<class ...Args>
		node_pointer insert_left_node(const node_pointer parent, Args&&... args)
		{
			node_pointer node = nullptr;
			if (parent != nullptr)
			{
				node = this->create_node(::std::forward<Args>(args)...);
				node->parent = parent;
				node->left = parent->left;
				node->right = nullptr;
				parent->left = node;
				if (header->left == parent)
					header->left = node;
				++count;
			}
			return node;
		}

		template<class ...Args>
		node_pointer insert_right_node(const node_pointer parent, Args&&... args)
		{
			node_pointer node = nullptr;
			if (parent != nullptr)
			{
				node = this->create_node(::std::forward<Args>(args)...);
				node->parent = parent;
				node->left = nullptr;
				node->right = parent->right;
				parent->right = node;
				if (header->right == parent)
					header->right = node;
				++count;
			}
			return node;
		}

		void copy_root(const node_pointer root)
		{
			node_pointer src = root;
			node_pointer dst = header;
			bin_tree_node_state state = bin_tree_state_root;
			node_pointer node = this->create_node(root->data);
			node->parent = dst;
			node->left = nullptr;
			node->right = nullptr;
			dst->parent = node;
			dst = node;
			do
			{
				if (state != bin_tree_state_parent)
				{
					if (cur->left != nullptr)
					{
						cur = cur->left;
						state = bin_tree_state_left;
						node = this->create_node(src->data);
						node->parent = dst;
						node->left = nullptr;
						node->right = nullptr;
						dst->left = node;
						dst = node;
					}
					else if (cur->right != nullptr)
					{
						cur = cur->right;
						state = bin_tree_state_right;
						node = this->create_node(src->data);
						node->parent = dst;
						node->left = nullptr;
						node->right = nullptr;
						dst->left = node;
						dst = node;
					}
					else if (src != src->parent->parent && src != src->parent->right)
					{
						src = src->parent->right;
						state = bin_tree_state_sibling;
						node = this->create_node(src->data);
						node->parent = dst->parent;
						node->left = nullptr;
						node->right = nullptr;
						dst->parent->right = node;
						dst = node;
					}
					else
					{
						cur = cur->parent;
						dst = dst->parent;
						state = bin_tree_state_parent;
					}
				}
				else
				{
					if (src != src->parent->parent && src != src->parent->right)
					{
						src = src->parent->right;
						state = bin_tree_state_sibling;
						node = this->create_node(src->data);
						node->parent = dst->parent;
						node->left = nullptr;
						node->right = nullptr;
						dst->parent->right = node;
						dst = node;
					}
					else
					{
						cur = cur->parent;
						dst = dst->parent;
					}
				}
			} while (cur != src);
			header->left = get_leftmost(header->parent);
			header->right = get_rightmost(header->parent);
		}

		void erase_children(const node_pointer parent)
		{
			if (parent->left == nullptr && parent->right == nullptr)
				return;
			node_pointer cur = parent;
			node_pointer next = nullptr;
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
					--count;
				}
			} while (cur != parent);
		}

		void erase_node(const node_pointer position)
		{
			if (position == position->parent->left)
				position->parent->left = nullptr;
			else
				position->parent->right = nullptr;
			erase_children(position);
			this->destroy_node(position);
			--count;
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
	private:
		node_pointer header;
		size_type    count;
	};

} // namespace core

#endif
