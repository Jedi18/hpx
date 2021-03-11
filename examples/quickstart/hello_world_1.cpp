//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to execute a HPX-thread printing
// "Hello World!" once. That's all.

//[hello_world_1_getting_started
// Including 'hpx/hpx_main.hpp' instead of the usual 'hpx/hpx_init.hpp' enables
// to use the plain C-main below as the direct main HPX entry point.
#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/algorithm.hpp>

static bool abs_compare(int a, int b)
{
    return (std::abs(a) < std::abs(b));
}

int main()
{
    // Min element
    std::vector<int> v{3, 1, 4, 1, 5, 9};

    std::vector<int>::iterator result = hpx::min_element(v.begin(), v.end());
    hpx::cout << "min element at: " << std::distance(v.begin(), result) << "\n";

    // Max element
    std::vector<int> v1{3, 1, -14, 1, 5, 9};
    std::vector<int>::iterator result1;

    result1 = hpx::max_element(v1.begin(), v1.end());
    hpx::cout << "max element at: " << std::distance(v1.begin(), result1) << '\n';

    result1 = hpx::max_element(hpx::execution::seq, v1.begin(), v1.end(), abs_compare);
    hpx::cout << "max element (absolute) at: "
              << std::distance(v1.begin(), result1) << '\n';

    const auto v2 = {3, 9, 1, 4, 2, 5, 9};
    const auto elem = hpx::minmax_element(begin(v2), end(v2));

    hpx::cout << "min = " << elem.min << ", max = " << elem.max << '\n';

    return 0;
}
//]
