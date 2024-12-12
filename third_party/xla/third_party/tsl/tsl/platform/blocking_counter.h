/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_TSL_PLATFORM_BLOCKING_COUNTER_H_
#define TENSORFLOW_TSL_PLATFORM_BLOCKING_COUNTER_H_

#include <atomic>
#include <chrono>  // NOLINT
#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tsl/platform/logging.h"

namespace tsl {

class BlockingCounter {
 public:
  BlockingCounter(int initial_count)
      : state_(initial_count << 1), notified_(false) {
    CHECK_GE(initial_count, 0);
    DCHECK_EQ((static_cast<unsigned int>(initial_count) << 1) >> 1,
              initial_count);
  }

  ~BlockingCounter() = default;

  static thread_local constexpr char kNonce = 0;

  inline void DecrementCount() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
      DCHECK_NE(((v + 2) & ~1), 0);
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    absl::MutexLock l(&mu_);
    DCHECK(!notified_);
    notified_ = true;
    cond_var_.SignalAll();
  }

  inline void Wait() {
    const char* last_waiter_addr =
        last_waiter_addr_.load(std::memory_order_relaxed);
    if (last_waiter_addr != nullptr) {
      CHECK_EQ(last_waiter_addr, &kNonce)
          << "multiple threads called WaitFor()";
    } else {
      if (!last_waiter_addr_.compare_exchange_strong(
              last_waiter_addr, &kNonce, std::memory_order_relaxed)) {
        LOG(FATAL) << "Got " << last_waiter_addr_ << " instead of " << -1;
      }
    }
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return;
    absl::MutexLock l(&mu_);

    // only one thread may call Wait(). To support more than one thread,
    // implement a counter num_to_exit, like in the Barrier class.
    CHECK_EQ(num_waiting_, 0) << "multiple threads called Wait()";
    num_waiting_++;

    while (!notified_) {
      cond_var_.Wait(&mu_);
    }
  }
  // Wait for the specified time, return false iff the count has not dropped to
  // zero before the timeout expired.
  inline bool WaitFor(std::chrono::milliseconds ms) {
    const char* last_waiter_addr =
        last_waiter_addr_.load(std::memory_order_relaxed);
    if (last_waiter_addr != nullptr) {
      CHECK_EQ(last_waiter_addr, &kNonce)
          << "multiple threads called WaitFor()";
    } else {
      if (!last_waiter_addr_.compare_exchange_strong(
              last_waiter_addr, &kNonce, std::memory_order_relaxed)) {
        LOG(FATAL) << "Got " << last_waiter_addr_ << " instead of " << -1;
      }
    }

    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return true;
    absl::Duration timeout = absl::FromChrono(ms);
    absl::MutexLock l(&mu_);

    // only one thread may call Wait(). To support more than one thread,
    // implement a counter num_to_exit, like in the Barrier class.

    while (!notified_) {
      if (cond_var_.WaitWithTimeout(&mu_, timeout)) {
        return false;
      }
    }
    return true;
  }

 private:
  absl::Mutex mu_;
  absl::CondVar cond_var_;
  std::atomic<int> state_;  // low bit is waiter flag
  std::atomic<const char*> last_waiter_addr_ = nullptr;
  int num_waiting_ ABSL_GUARDED_BY(mu_) = 0;
  bool notified_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_BLOCKING_COUNTER_H_
