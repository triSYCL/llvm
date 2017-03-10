//===- llvm/SYCL.h - SYCL related declarations ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares some common SYCL things.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SYCL_H
#define LLVM_SYCL_H

#include <cstddef>
#include "llvm/IR/Function.h"

namespace llvm {
namespace sycl {

/// Test if a function is a SYCL kernel
bool isKernel(const Function &F);

/// This is a llvm local version of __cxa_demangle. Other than the name and
/// being in the llvm namespace it is identical.
///
/// This is a back-port from LLVM 4.0
///
/// The mangled_name is demangled into buf and returned. If the buffer is not
/// large enough, realloc is used to expand it.
///
/// The *status will be set to
///   unknown_error: -4
///   invalid_args:  -3
///   invalid_mangled_name: -2
///   memory_alloc_failure: -1
///   success: 0
char *itaniumDemangle(const char *mangled_name, char *buf, size_t *n,
                      int *status);

}
}

#endif
