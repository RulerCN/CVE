// CVE.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <iomanip>

#include "core\cpu\cpu_convert.h"
#include "core\cpu\cpu_convert_scale.h"
#include "core\cpu\cpu_reduce.h"
#include "core\cpu\cpu_transpose.h"
#include "image\bitmap.h"
#include "ann\mnist.h"

// Print vector
template<class Allocator>
void print(const char *name, const core::vector<signed char, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<unsigned char, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<signed short, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<unsigned short, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<int, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<float, Allocator> &vec);
// Print matrix
template<class Allocator>
void print(const char *name, const core::matrix<signed char, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned char, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<signed short, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned short, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<int, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<float, Allocator> &mat);

int main()
{
	core::global::enable_simd(true);

	//core::matrix<unsigned char> output;
	//img::bitmap_palette palette;
	//img::bitmap::decode("data/train/1.bmp", output, palette);
	//return 0;

	const size_t batch     = 10;
	const size_t rows      = 28;
	const size_t columns   = 28;
	const size_t length    = 10;
	const size_t dimension = 1;

	ann::mnist<> mnist("data/mnist");
	core::tensor<float> train_images_flt(batch, rows, columns, dimension);
	core::tensor<unsigned char> train_images(batch, rows, columns, dimension);
	core::vector<unsigned char> train_labels(length, dimension);
	core::tensor<unsigned char> test_images(batch, rows, columns, dimension);
	core::vector<unsigned char> test_labels(length, dimension);

	mnist.train.shuffle(1U);
	mnist.train.next_batch(train_images_flt, train_labels);
	core::cpu_convert(train_images, train_images_flt);

	img::bitmap::encode("data/train/1.bmp", train_images[0]);
	img::bitmap::encode("data/train/2.bmp", train_images[1]);
	img::bitmap::encode("data/train/3.bmp", train_images[2]);
	img::bitmap::encode("data/train/4.bmp", train_images[3]);
	img::bitmap::encode("data/train/5.bmp", train_images[4]);
	img::bitmap::encode("data/train/6.bmp", train_images[5]);
	img::bitmap::encode("data/train/7.bmp", train_images[6]);
	img::bitmap::encode("data/train/8.bmp", train_images[7]);

	mnist.test.shuffle(1U);
	mnist.test.next_batch(test_images, test_labels);
	img::bitmap::encode("data/test/1.bmp", test_images[0]);
	img::bitmap::encode("data/test/2.bmp", test_images[1]);
	img::bitmap::encode("data/test/3.bmp", test_images[2]);
	img::bitmap::encode("data/test/4.bmp", test_images[3]);
	img::bitmap::encode("data/test/5.bmp", test_images[4]);
	img::bitmap::encode("data/test/6.bmp", test_images[5]);
	img::bitmap::encode("data/test/7.bmp", test_images[6]);
	img::bitmap::encode("data/test/8.bmp", test_images[7]);

	//size_t row = 13;
	//size_t col = 17;
	//size_t dim = 1;
	//core::matrix<signed char> x(row, col, dim);
	//core::matrix<signed char> t(col, row, dim);
	//core::vector<signed char> col_min(col, dim, static_cast<signed char>(core::int8_max));
	//core::vector<signed char> col_max(col, dim, static_cast<signed char>(core::int8_min));
	//core::vector<signed int> col_sum(col, dim, static_cast<signed int>(core::int32_zero));
	//core::vector<signed char> row_min(row, dim, static_cast<signed char>(core::int8_max));
	//core::vector<signed char> row_max(row, dim, static_cast<signed char>(core::int8_min));
	//core::vector<signed int> row_sum(row, dim, static_cast<signed int>(core::int32_zero));
	//// Initialization matrix
	//x.linear_fill(static_cast<signed char>(1), static_cast<signed char>(2), static_cast<signed char>(1));
	//// Matrix operation
	//core::transpose(t, x);
	//core::reduce(col_min, x, core::rm_col_min);
	//core::reduce(col_max, x, core::rm_col_max);
	//core::reduce(col_sum, x, core::rm_col_sum);
	//core::reduce(row_min, x, core::rm_row_min);
	//core::reduce(row_max, x, core::rm_row_max);
	//core::reduce(row_sum, x, core::rm_row_sum);
	//print("X", x);
	//print("T", t);
	//print("col_min", col_min);
	//print("col_max", col_max);
	//print("col_sum", col_sum);
	//print("row_min", row_min);
	//print("row_max", row_max);
	//print("row_sum", row_sum);
	return 0;
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed char, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(static_cast<unsigned char>(*i)) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<unsigned char, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*i) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed short, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(static_cast<unsigned short>(*i)) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<unsigned short, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(*i) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed int, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(8) << *i << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<float, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << std::setfill('0') << std::setw(8) << *i << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<signed char, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(static_cast<unsigned char>(*i)) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned char, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*i) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<signed short, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(static_cast<unsigned short>(*i)) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned short, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(*i) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<int, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(8) << *i << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<float, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::setfill('0') << std::setw(8) << *i << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}
