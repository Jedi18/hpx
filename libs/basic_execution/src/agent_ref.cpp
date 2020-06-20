//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/basic_execution/agent_ref.hpp>
#include <hpx/modules/assertion.hpp>
#ifdef HPX_HAVE_VERIFY_LOCKS
#include <hpx/basic_execution/register_locks.hpp>
#endif
#include <hpx/basic_execution/this_thread.hpp>
#include <hpx/modules/format.hpp>

#include <cstddef>
#include <iostream>

namespace hpx { namespace basic_execution {

    void agent_ref::yield(const char* desc)
    {
        HPX_ASSERT(*this == hpx::basic_execution::this_thread::agent());
        // verify that there are no more registered locks for this OS-thread
#ifdef HPX_HAVE_VERIFY_LOCKS
        util::verify_no_locks();
#endif
        impl_->yield(desc);
    }

    void agent_ref::yield_k(std::size_t k, const char* desc)
    {
        HPX_ASSERT(*this == hpx::basic_execution::this_thread::agent());
        // verify that there are no more registered locks for this OS-thread
#ifdef HPX_HAVE_VERIFY_LOCKS
        util::verify_no_locks();
#endif
        impl_->yield(desc);
    }

    void agent_ref::suspend(const char* desc)
    {
        HPX_ASSERT(*this == hpx::basic_execution::this_thread::agent());
        // verify that there are no more registered locks for this OS-thread
#ifdef HPX_HAVE_VERIFY_LOCKS
        util::verify_no_locks();
#endif
        impl_->suspend(desc);
    }

    void agent_ref::resume(const char* desc)
    {
        HPX_ASSERT(*this != hpx::basic_execution::this_thread::agent());
        impl_->resume(desc);
    }

    void agent_ref::abort(const char* desc)
    {
        HPX_ASSERT(*this != hpx::basic_execution::this_thread::agent());
        impl_->abort(desc);
    }

    void agent_ref::sleep_for(
        hpx::util::steady_duration const& sleep_duration, const char* desc)
    {
        HPX_ASSERT(*this == hpx::basic_execution::this_thread::agent());
        impl_->sleep_for(sleep_duration, desc);
    }

    void agent_ref::sleep_until(
        hpx::util::steady_time_point const& sleep_time, const char* desc)
    {
        HPX_ASSERT(*this == hpx::basic_execution::this_thread::agent());
        impl_->sleep_until(sleep_time, desc);
    }

    std::ostream& operator<<(std::ostream& os, agent_ref const& a)
    {
        hpx::util::format_to(os, "agent_ref{{{}}}", a.impl_->description());
        return os;
    }
}}    // namespace hpx::basic_execution