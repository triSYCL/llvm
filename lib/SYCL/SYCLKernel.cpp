//===- SYCLKernel.cpp                                       ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect SYCL kernels
// ===------------------------------------------------------------------- -===//

#include <cstdlib>
#include "llvm/ADT/StringRef.h"
// Wait for LLVM 4.0...
// #include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/SYCL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace sycl {

/// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
/// option -debug or -debug-only=SYCL
#define DEBUG_TYPE "SYCL"

/// The function template used to instantiate a kernel inside triSYCL
/// is used as marker to detect the kernel functions
StringRef SYCLKernelPrefix { "void cl::sycl::detail::instantiate_kernel<" };

/// Test if a function is a SYCL kernel
bool isKernel(const Function &F) {
  if (!F.hasName())
    return false;

  bool KernelFound = false;
  // Demangle C++ name for human beings
  int Status;
  char *Demangled = itaniumDemangle(F.getName().str().c_str(),
                                    nullptr,
                                    nullptr,
                                    &Status);
  if (Demangled) {
    DEBUG(errs() << " Demangled: " << Demangled << "\n");
    // A kernel is just a function starting with the well known name
    if (StringRef { Demangled }.startswith(SYCLKernelPrefix)) {
      DEBUG(errs() << "\n\tKernel found!\n\n");
      KernelFound = true;
    }
  }
  free(Demangled);
  return KernelFound;
}

}
}
