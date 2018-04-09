// CVE.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include "core\cpu\convert.h"
#include "core\cpu\convert_scale.h"
#include "core\cpu\reduce.h"
#include "core\cpu\transpose.h"

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
	size_t row = 13;
	size_t col = 17;
	size_t dim = 1;
	core::matrix<signed char> x(row, col, dim);
	core::matrix<signed char> t(col, row, dim);
	core::vector<signed char> col_min(col, dim, static_cast<signed char>(core::int8_max));
	core::vector<signed char> col_max(col, dim, static_cast<signed char>(core::int8_min));
	core::vector<signed int> col_sum(col, dim, static_cast<signed int>(core::int32_zero));
	core::vector<signed char> row_min(row, dim, static_cast<signed char>(core::int8_max));
	core::vector<signed char> row_max(row, dim, static_cast<signed char>(core::int8_min));
	core::vector<signed int> row_sum(row, dim, static_cast<signed int>(core::int32_zero));
	// Initialization matrix
	x.linear_fill(static_cast<signed char>(1), static_cast<signed char>(2), static_cast<signed char>(1));
	// Matrix operation
	core::transpose(t, x);
	core::reduce(col_min, x, core::rm_col_min);
	core::reduce(col_max, x, core::rm_col_max);
	core::reduce(col_sum, x, core::rm_col_sum);
	core::reduce(row_min, x, core::rm_row_min);
	core::reduce(row_max, x, core::rm_row_max);
	core::reduce(row_sum, x, core::rm_row_sum);
	print("X", x);
	print("T", t);
	print("col_min", col_min);
	print("col_max", col_max);
	print("col_sum", col_sum);
	print("row_min", row_min);
	print("row_max", row_max);
	print("row_sum", row_sum);
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
