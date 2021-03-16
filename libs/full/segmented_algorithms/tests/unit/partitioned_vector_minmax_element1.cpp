//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_minmax.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
#define SIZE 64

template <typename T>
void initialize(hpx::partitioned_vector<T>& xvalues)
{
    T init_array[SIZE] = {1, 2, 3, 4, 5, 1, 2, 3, 3, 5, 5, 3, 4, 2, 3, 2, 1, 2,
        3, 4, 5, 6, 5, 6, 1, 2, 3, 4, 1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 2, 3, 4,
        1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 7, 6, 5, 7, 5, 4, 2, 3, 4, 5, 2};
    for (int i = 0; i < SIZE; i++)
    {
        xvalues.set_value(i, init_array[i]);
    }
}

template <typename ExPolicy, typename T, typename Func>
void test_minmax_element(ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues,
    Func&& f, T expected_result_min, T expected_result_max)
{
    auto result =
        hpx::minmax_element(policy, xvalues.begin(), xvalues.end(), f);
    HPX_TEST_EQ(*result.min, expected_result_min);
    HPX_TEST_EQ(*result.max, expected_result_max);
}

template <typename ExPolicy, typename T, typename Func>
void test_minmax_element_async(ExPolicy&& policy,
    hpx::partitioned_vector<T>& xvalues, Func&& f, T expected_result_min,
    T expected_result_max)
{
    auto result =
        hpx::minmax_element(policy, xvalues.begin(), xvalues.end(), f).get();
    HPX_TEST_EQ(*result.min, expected_result_min);
    HPX_TEST_EQ(*result.max, expected_result_max);
}

template <typename T>
void minmax_element_tests(std::vector<hpx::id_type>& localities)
{
    hpx::partitioned_vector<T> xvalues(
        SIZE, T(0), hpx::container_layout(localities));
    initialize(xvalues);

    test_minmax_element(
        hpx::execution::seq, xvalues, std::less<int>(), T(1), T(7));
    /*test_minmax_element(
        hpx::execution::par, xvalues, std::less<int>(), T(1), T(7));
    test_minmax_element_async(hpx::execution::seq(hpx::execution::task),
        xvalues, std::less<int>(), T(1), T(7));
    test_minmax_element_async(hpx::execution::par(hpx::execution::task),
        xvalues, std::less<int>(), T(1), T(7));

    test_minmax_element(
        hpx::execution::seq, xvalues, std::greater<int>(), T(7), T(1));
    test_minmax_element(
        hpx::execution::par, xvalues, std::greater<int>(), T(7), T(1));
    test_minmax_element_async(hpx::execution::seq(hpx::execution::task),
        xvalues, std::greater<int>(), T(7), T(1));
    test_minmax_element_async(hpx::execution::par(hpx::execution::task),
        xvalues, std::greater<int>(), T(7), T(1));*/
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    minmax_element_tests<int>(localities);
    return hpx::util::report_errors();
}
#endif
