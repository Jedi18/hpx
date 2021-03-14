//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/minmax.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {

    template <typename T>
    using minmax_element_result = hpx::parallel::util::min_max_result<T>;

    ///////////////////////////////////////////////////////////////////////////
    // segmented_minmax
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_minormax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<SegIter> positions;

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.push_back(traits::compose(send, out));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.push_back(traits::compose(sit, out));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        local_iterator_type out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f, proj);

                        positions.push_back(traits::compose(sit, out));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.push_back(traits::compose(sit, out));
                }
            }

            return Algo::sequential_minmax_element_ind(
                policy, positions.begin(), positions.size(), f, proj);
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_minormax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<SegIter>::value>
                forced_seq;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<SegIter>> segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<SegIter>(
                        dispatch_async(traits::get_id(sit), algo, policy,
                            forced_seq(), beg, end, f, proj),
                        [send](local_iterator_type const& out) -> SegIter {
                            return traits::compose(send, out);
                        }));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<SegIter>(
                        dispatch_async(traits::get_id(sit), algo, policy,
                            forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_type const& out) -> SegIter {
                            return traits::compose(sit, out);
                        }));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(hpx::make_future<SegIter>(
                            dispatch_async(traits::get_id(sit), algo, policy,
                                forced_seq(), beg, end, f, proj),
                            [sit](local_iterator_type const& out) -> SegIter {
                                return traits::compose(sit, out);
                            }));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<SegIter>(
                        dispatch_async(traits::get_id(sit), algo, policy,
                            forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_type const& out) -> SegIter {
                            return traits::compose(sit, out);
                        }));
                }
            }

            return result::get(dataflow(
                [=](std::vector<hpx::future<SegIter>>&& r) -> SegIter {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);

                    std::vector<SegIter> res = hpx::util::unwrap(std::move(r));
                    return Algo::sequential_minmax_element_ind(
                        policy, res.begin(), res.size(), f, proj);
                },
                std::move(segments)));
        }
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // segmented_minmax
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy,
            minmax_element_result<SegIter>>::type
        segmented_minmax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef minmax_element_result<SegIter> result_type;

            typedef std::pair<local_iterator_type, local_iterator_type>
                local_iterator_pair_type;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<result_type> positions;

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    result_type res = {traits::compose(send, out.first),
                        traits::compose(send, out.second)};
                    positions.push_back(res);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    result_type res = {traits::compose(sit, out.first),
                        traits::compose(sit, out.second)};
                    positions.push_back(res);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        local_iterator_pair_type out =
                            dispatch(traits::get_id(sit), algo, policy,
                                std::true_type(), beg, end, f, proj);

                        result_type res = {traits::compose(sit, out.first),
                            traits::compose(sit, out.second)};
                        positions.push_back(res);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    result_type res = {traits::compose(sit, out.first),
                        traits::compose(sit, out.second)};
                    positions.push_back(res);
                }
            }

            return Algo::sequential_minmax_element_ind(
                policy, positions.begin(), positions.size(), f, proj);
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy,
            minmax_element_result<SegIter>>::type
        segmented_minmax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<SegIter>::value>
                forced_seq;
            typedef minmax_element_result<SegIter> result_type;
            typedef util::detail::algorithm_result<ExPolicy, result_type>
                result;

            typedef std::pair<local_iterator_type, local_iterator_type>
                local_iterator_pair_type;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<result_type>> segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<result_type>(
                        dispatch_async(traits::get_id(sit), algo, policy,
                            forced_seq(), beg, end, f, proj),
                        [send](local_iterator_pair_type out) -> result_type {
                            result_type res = {traits::compose(send, out.first),
                                traits::compose(send, out.second)};
                            return res;
                        }));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<result_type>(
                        dispatch_async(traits::get_id(sit), algo, policy,
                            forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_pair_type const& out)
                            -> result_type {
                            result_type res = {traits::compose(sit, out.first),
                                traits::compose(sit, out.second)};
                            return res;
                        }));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(hpx::make_future<result_type>(
                            dispatch_async(traits::get_id(sit), algo, policy,
                                forced_seq(), beg, end, f, proj),
                            [sit](local_iterator_pair_type const& out)
                                -> result_type {
                                result_type res = {
                                    traits::compose(sit, out.first),
                                    traits::compose(sit, out.second)};
                                return res;
                            }));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<result_type>(
                        dispatch_async(traits::get_id(sit), algo, policy,
                            forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_pair_type const& out)
                            -> result_type {
                            result_type res = {traits::compose(sit, out.first),
                                traits::compose(sit, out.second)};
                            return res;
                        }));
                }
            }

            return result::get(dataflow(
                [=](std::vector<hpx::future<result_type>>&& r) -> result_type {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);

                    std::vector<result_type> res =
                        hpx::util::unwrap(std::move(r));
                    return Algo::sequential_minmax_element_ind(
                        policy, res.begin(), res.size(), f, proj);
                },
                std::move(segments)));
        }
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx { namespace segmented {

    template <typename T>
    using minmax_element_result = hpx::parallel::util::min_max_result<T>;

    // clang-format off
    template <typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    SegIter tag_invoke(hpx::min_element_t, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        if (first == last || ++first == last)
        {
            return first;
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::v1::detail::segmented_minormax(
            hpx::parallel::v1::detail::min_element<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, std::forward<F>(f),
            hpx::parallel::util::projection_identity{}, std::true_type{});
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        SegIter>::type
    tag_invoke(hpx::min_element_t, ExPolicy&& policy, SegIter first,
        SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (first == last || ++first == last)
        {
            return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                SegIter>::get(std::move(first));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::v1::detail::segmented_minormax(
            hpx::parallel::v1::detail::min_element<
                typename iterator_traits::local_iterator>(),
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            hpx::parallel::util::projection_identity{}, is_seq());
    }

    // clang-format off
    template <typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    SegIter tag_invoke(hpx::max_element_t, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        if (first == last || ++first == last)
        {
            return first;
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::v1::detail::segmented_minormax(
            hpx::parallel::v1::detail::max_element<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, std::forward<F>(f),
            hpx::parallel::util::projection_identity{}, std::true_type{});
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        SegIter>::type
    tag_invoke(hpx::max_element_t, ExPolicy&& policy, SegIter first,
        SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (first == last || ++first == last)
        {
            return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                SegIter>::get(std::move(first));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::v1::detail::segmented_minormax(
            hpx::parallel::v1::detail::max_element<
                typename iterator_traits::local_iterator>(),
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            hpx::parallel::util::projection_identity{}, is_seq());
    }

    // clang-format off
    template <typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    minmax_element_result<SegIter> tag_invoke(
        hpx::minmax_element_t, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        if (first == last || ++first == last)
        {
            return {first, first};
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::v1::detail::segmented_minmax(
            hpx::parallel::v1::detail::minmax_element<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, std::forward<F>(f),
            hpx::parallel::util::projection_identity{}, std::true_type{});
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        minmax_element_result<SegIter>>::type
    tag_invoke(hpx::minmax_element_t, ExPolicy&& policy, SegIter first,
        SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
        using result_type = minmax_element_result<SegIter>;

        if (first == last || ++first == last)
        {
            result_type result = {first, first};
            return hpx::parallel::util::detail::algorithm_result<ExPolicy, result_type>::get(
                std::move(result));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::v1::detail::segmented_minmax(
            hpx::parallel::v1::detail::minmax_element<
                typename iterator_traits::local_iterator>(),
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            hpx::parallel::util::projection_identity{}, is_seq());
    }
}}    // namespace hpx::segmented
