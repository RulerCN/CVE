// CVE.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "core\uninitialized.h"

int main()
{
	int a[100], b[100];
	core::uninitialized_default_construct_n(a, 100);
	core::uninitialized_value_construct_n(b, 100);

	float aa = int();
	float bb;

    return 0;
}

