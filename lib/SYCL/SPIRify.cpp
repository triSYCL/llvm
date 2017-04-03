//===- SPIRify.cpp                           ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Rewrite the kernels and functions so that they are compatible with SPIR
// ===---------------------------------------------------------------------===//

#include <cstddef>

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/SYCL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
/// option -debug or -debug-only=SYCL
#define DEBUG_TYPE "SYCL"


/// Displayed with -stats
STATISTIC(SYCLKernelProcessed, "Number of SYCL kernel functions processed");

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Replace a SYCL kernel code by a function serializing its arguments
struct SPIRify : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  SPIRify() : ModulePass(ID) {}


  bool doInitialization(Module &M) override {
    DEBUG(errs() << "Enter: " << M.getModuleIdentifier() << "\n\n");

    // Do not change the code
    return false;
  }


  bool doFinalization(Module &M) override {
    DEBUG(errs() << "Exit: " << M.getModuleIdentifier() << "\n\n");

    // Do not change the code
    return false;
  }


  /// Replace the kernel instructions by the serialization of its arguments
  void kernelSPIRify(Function &F) {
    ++SYCLKernelProcessed;

    // Move to SPIR kernel
    F.setCallingConv(CallingConv::SPIR_KERNEL);
  }


  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    for (auto &F : M.functions()) {
      // Only consider definition of SYCL kernels
      // \todo Put SPIR calling convention on declarations too
      if (!F.isDeclaration() && sycl::isKernel(F))
          kernelSPIRify(F);
    }

    // The module probably changed
    return true;
  }
};

}

char SPIRify::ID = 0;
static RegisterPass<SPIRify> X {
  "SPIRify",
  "pass to make functions and kernels SPIR compatible"
 };
