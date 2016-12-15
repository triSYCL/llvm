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

#include "llvm/IR/Function.h"

namespace llvm {
namespace sycl {

/// Test if a function is a SYCL kernel
bool isKernel(const Function &F);

}
}

#endif
