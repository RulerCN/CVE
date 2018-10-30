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

#ifndef __NN_MNIST_H__
#define __NN_MNIST_H__

#include "mnist_data.h"

namespace nn
{
	// Class template mnist
	template <class Allocator = ::core::allocator<unsigned char> >
	class mnist
	{
	public:
		// construct/copy/destroy:

		mnist(void)
		{}
		mnist(const ::std::string& folder)
		{
			load(folder);
		}
		// Not recommended. (Olny provided for Microsoft Visual Studio)
#		if defined(_MSC_VER)
		mnist(const ::std::wstring& folder)
		{
			load(folder);
		}
#		endif
		~mnist(void)
		{}

		// Load MNIST database

		bool load(const ::std::string& folder)
		{
			::std::string train_labels = folder + "/train-labels.idx1-ubyte";
			::std::string train_images = folder + "/train-images.idx3-ubyte";
			::std::string test_labels = folder + "/t10k-labels.idx1-ubyte";
			::std::string test_images = folder + "/t10k-images.idx3-ubyte";
			return (train.load(train_images, train_labels) && test.load(test_images, test_labels));
		}

		// Not recommended. (Olny provided for Microsoft Visual Studio)
#		if defined(_MSC_VER)
		bool load(const ::std::wstring& folder)
		{
			::std::wstring train_labels = folder + L"/train-labels.idx1-ubyte";
			::std::wstring train_images = folder + L"/train-images.idx3-ubyte";
			::std::wstring test_labels = folder + L"/t10k-labels.idx1-ubyte";
			::std::wstring test_images = folder + L"/t10k-images.idx3-ubyte";
			return (train.load(train_images, train_labels) && test.load(test_images, test_labels));
		}
#		endif
	public:
		mnist_data<Allocator> train;
		mnist_data<Allocator> test;
	};

} // namespace nn

#endif
