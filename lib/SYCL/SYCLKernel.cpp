//===- SYCLKernel.cpp                                       ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect SYCL kernels and rename kernel too shorten unique names
// ===------------------------------------------------------------------- -===//

#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>

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

/// For some implementations, rename the kernels with shorter and cleaner names
/// starting with this prefix
std::string SYCLKernelShortPrefix { "TRISYCL_kernel_" };

/// Test if a function is a SYCL kernel
bool isKernel(const Function &F) {
  auto lookForKernel = [&] {
    if (!F.hasName())
      return false;

    if (F.getName().startswith(SYCLKernelShortPrefix))
      return true;

    // Demangle C++ name for human beings
    int Status;
    // Use a \c unique_ptr to deallocate memory on exit
    std::unique_ptr<const char> Demangled {
      itaniumDemangle(F.getName().str().c_str(),
                      nullptr,
                      nullptr,
                      &Status) };
    if (Demangled) {
      DEBUG(errs() << " Demangled: " << Demangled.get() << "\n");
      // A kernel is just a function starting with the well known name
      return StringRef { Demangled.get() }.startswith(SYCLKernelPrefix);
    }
    return false;
  };

  auto KernelFound = lookForKernel();
  DEBUG(if (KernelFound)
          errs() << "\n\tKernel found!\n\n");

  return KernelFound;
}


/// Mapping from the full kernel mangle named to a unique integer ID
std::map<std::string, std::size_t> SimplerKernelNames;

/// The kernel ID to use, counting from 0
std::size_t NextKernelID = 0;


/// Register a kernel with its full name and returns its ID
///
/// If the kernel is already registered, do not register it again.
std::size_t registerSYCLKernel(const std::string &LongKernelName) {
  auto Translation = SimplerKernelNames.emplace(LongKernelName, NextKernelID);
  /// If a new kernel has been registered, then compute the next ID to use
  if (Translation.second)
    ++NextKernelID;
  // Return the ID for the kernel
  return Translation.first->second;
}


/// Construct a kernel short name for an ID
std::string constructSYCLKernelShortName(std::size_t Id) {
  std::stringstream S;
  S << SYCLKernelShortPrefix << Id;
  return S.str();
}


/// Register a kernel with its full name and returns its short name
///
/// If the kernel is already registered, do not register it again.
std::string
registerSYCLKernelAndGetShortName(const std::string &LongKernelName) {
  return constructSYCLKernelShortName(registerSYCLKernel(LongKernelName));
}

}
}
