// CVE.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include "core\cpu\convert.h"
#include "core\cpu\convert_scale.h"
#include "core\cpu\reduce.h"

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

	// reduce (signed int)
	size_t row = 13;
	size_t col = 17;
	size_t dim = 1;
	core::matrix<signed int> x(row, col, dim);
	//core::matrix<signed int> t(col, row, dim);
	core::vector<signed int> col_min(col, dim, static_cast<signed int>(core::int8_max));
	core::vector<signed int> col_max(col, dim, static_cast<signed int>(core::int8_min));
	core::vector<signed int> col_sum(col, dim, static_cast<signed int>(core::int32_zero));
	//core::vector<signed int> row_min(row, dim, static_cast<signed int>(core::int8_max));
	//core::vector<signed int> row_max(row, dim, static_cast<signed int>(core::int8_min));
	//core::vector<signed int> row_sum(row, dim, static_cast<signed int>(core::int32_zero));
	// Initialization matrix
	x.linear_fill(static_cast<signed int>(1), static_cast<signed int>(2), static_cast<signed int>(1));
	// Matrix operation
	//core::transpose(t, x);
	core::reduce(col_min, x, core::rm_col_min);
	core::reduce(col_max, x, core::rm_col_max);
	core::reduce(col_sum, x, core::rm_col_sum);
	//core::reduce(row_min, x, core::rm_row_min);
	//core::reduce(row_max, x, core::rm_row_max);
	//core::reduce(row_sum, x, core::rm_row_sum);
	print("col_min", col_min);
	print("col_max", col_max);
	print("col_sum", col_sum);


	//size_t row = 20;
	//size_t col = 30;
	//size_t dim = 1;
	//core::matrix<unsigned char> a(row, col, dim);
	//core::matrix<float> b(row, col, dim);
	//// Initialization matrix
	//a.linear_fill(1, 1);
	//core::convert_scale(b, a, 1.0f/255.0f);
	////print("A", a);
	//print("B", b);

    return 0;
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
			std::cout << *i << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}