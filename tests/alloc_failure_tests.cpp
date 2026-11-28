// Copyright © 2026 Apple Inc.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dlfcn.h>
#include <objc/runtime.h>

#include <cstdlib>
#include <new>
#include <utility>

#include "doctest/doctest.h"

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

// Fail one allocation made by this binary or by MLX, which may be a shared
// library.
thread_local long countdown = 0;
thread_local bool fired = false;

const void* image_of(const void* address) {
  Dl_info info{};
  return dladdr(address, &info) ? info.dli_fbase : nullptr;
}

const void* const test_image =
    image_of(reinterpret_cast<const void*>(&image_of));
const void* const mlx_image = image_of(
    reinterpret_cast<const void*>(
        static_cast<void (*)(std::vector<array>)>(&async_eval)));

bool from_test_or_mlx(const void* caller) {
  auto image = image_of(caller);
  return image && (image == test_image || image == mlx_image);
}

void* allocate(std::size_t n, const void* caller) {
  if (countdown > 0 && from_test_or_mlx(caller) && --countdown == 0) {
    fired = true;
    throw std::bad_alloc();
  }
  if (auto* p = std::malloc(n ? n : 1)) {
    return p;
  }
  throw std::bad_alloc();
}

} // namespace

void* operator new(std::size_t n) {
  return allocate(n, __builtin_return_address(0));
}
void* operator new[](std::size_t n) {
  return allocate(n, __builtin_return_address(0));
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  std::free(p);
}

namespace {

array decode_step(array& cache, const array& w, int pos) {
  auto kv = ones({1, 4, 1, 64});
  cache = slice_update(cache, kv, {0, 0, pos, 0}, {1, 4, pos + 1, 64});
  return matmul(cache, w);
}

} // namespace

TEST_CASE("an allocation failure in async_eval leaves the stream usable") {
  auto w = ones({64, 64});
  for (long n = 1;; ++n) {
    CAPTURE(n);
    auto cache = zeros({1, 4, 16, 64});
    eval(cache, w);
    auto out = decode_step(cache, w, 1);

    fired = false;
    countdown = n;
    try {
      async_eval({out, cache});
    } catch (const std::bad_alloc&) {
    }
    countdown = 0;

    auto next = decode_step(cache, w, 2);
    eval(next);
    CHECK_EQ(sum(next).item<float>(), 2 * 4 * 64 * 64);
    if (!fired) {
      break;
    }
  }
}

namespace {

// Make the next commandBufferWithUnretainedReferences fail once.
enum class Failure { none, throws, returns_nil };
Failure next_buffer_failure = Failure::none;
IMP original_new_buffer = nullptr;

id failing_new_buffer(id self, SEL sel) {
  auto failure = std::exchange(next_buffer_failure, Failure::none);
  if (failure == Failure::throws) {
    throw std::bad_alloc();
  }
  if (failure == Failure::returns_nil) {
    return nil;
  }
  return reinterpret_cast<id (*)(id, SEL)>(original_new_buffer)(self, sel);
}

struct FailingNewBuffer {
  Method method;
  FailingNewBuffer() {
    auto queue = NS::TransferPtr(
        metal::device(Device::gpu).mtl_device()->newCommandQueue());
    method = class_getInstanceMethod(
        object_getClass(reinterpret_cast<id>(queue.get())),
        sel_registerName("commandBufferWithUnretainedReferences"));
    REQUIRE(method);
    original_new_buffer = method_setImplementation(
        method, reinterpret_cast<IMP>(&failing_new_buffer));
  }
  ~FailingNewBuffer() {
    method_setImplementation(method, original_new_buffer);
  }
};

} // namespace

TEST_CASE("failing to make a command buffer leaves the stream usable") {
  auto w = ones({64, 64});
  eval(w);
  FailingNewBuffer failing;

  for (auto failure : {Failure::throws, Failure::returns_nil}) {
    CAPTURE(static_cast<int>(failure));
    auto cache = zeros({1, 4, 16, 64});
    eval(cache);
    auto out = decode_step(cache, w, 1);
    next_buffer_failure = failure;
    CHECK_THROWS(eval(out));
    next_buffer_failure = Failure::none;

    auto next = decode_step(cache, w, 2);
    eval(next);
    CHECK_EQ(sum(next).item<float>(), 2 * 4 * 64 * 64);
  }
}
