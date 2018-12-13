//===- SYCLKernel.cpp                                       ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect SYCL kernels and rename kernel to shorten unique names
//
// Detect if functions have kernel as an ancestor
// ===------------------------------------------------------------------- -===//

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
// Wait for LLVM 4.0...
// #include "llvm/Demangle/Demangle.h"
#include "llvm/IR/CallSite.h"
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
      LLVM_DEBUG(errs() << " Demangled: " << Demangled.get() << "\n");
      // A kernel is just a function starting with the well known name
      return StringRef { Demangled.get() }.startswith(SYCLKernelPrefix);
    }
    return false;
  };

  auto KernelFound = lookForKernel();
  LLVM_DEBUG(if (KernelFound)
          errs() << "\n\tKernel found!\n\n");

  return KernelFound;
}

/// Test if functions having the kernel as an ancestor
bool isTransitivelyCalledFromKernel(Function &F,
                                    SmallPtrSet<Function *, 32> &FunctionsCalledByKernel) {
  for (auto &U : F.uses()) {
    CallSite CS{U.getUser()};
    if (auto I = CS.getInstruction()) {
      auto parent = I->getParent()->getParent();
      return FunctionsCalledByKernel.count(parent);
    }
  }

  return false;
}

/// Add the functions that are transitively called from the kernel in the set
void recordFunctionsCalledByKernel(CallGraphSCC &SCC, CallGraph &CG,
                                   SmallPtrSet<Function *, 32> &FunctionsCalledByKernel) {
  // Find the CallGraphNode that the kernel function belongs to.
  // Then, DFS algorithm starts from the kernel function CallGraphNode to
  // discover all the functions that have kernel as an ancestor and add them to
  // the FunctionsCalledByKernel set.
  // \note for (CallGraphNode *I : SCC) {} will run in bottom-up order
  for (auto SCCI = scc_begin(&CG); !SCCI.isAtEnd(); ++SCCI) {
    auto const &nextSCC = *SCCI;
    for (auto I : nextSCC)
      if(auto *F = I->getFunction())
        if (isKernel(*F) || isTransitivelyCalledFromKernel(*F, FunctionsCalledByKernel))
          FunctionsCalledByKernel.insert(F);
  }
}

/// Update the FunctionsCalledByKernel set when new CallGraphNode created in
/// CallGraph
void updateFunctionsCalledByKernel (CallGraphNode &NewNode,
                                    SmallPtrSet<Function *, 32> &FunctionsCalledByKernel) {
  auto *F = NewNode.getFunction();
  if (isTransitivelyCalledFromKernel(*F, FunctionsCalledByKernel))
    FunctionsCalledByKernel.insert(F);
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
