//===- SYCLSerializeArguments.cpp                           ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Replace a SYCL kernel code by a function serializing its arguments
// ===---------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/SYCL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CtorUtils.h"

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
struct SYCLSerializeArguments : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  SYCLSerializeArguments() : ModulePass(ID) {}


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
  void serializeKernelArguments(Function &F) {
    ++SYCLKernelProcessed;
    // Remove the code of the kernel first
    F.dropAllReferences();
    assert(F.empty() && "There should be no basic block left");

    // Insert the serialization code in its own basic block
    BasicBlock * BB = BasicBlock::Create(F.getContext(),
                                         "Serialize",
                                         &F);
    // Use an IRBuilder to ease IR creation in the basic block
    IRBuilder<> Builder(BB);

    for (Argument &A : F.args()) {
      DEBUG(dbgs() << "Serializing '" << A.getName() << "'.\n");
    }

    /* Add a "ret void" as the function terminator.
       Assume the return type of a kernel is void */
    Builder.CreateRetVoid();
  }


  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    for (auto &F : M.functions()) {
      DEBUG(errs() << "Function: ";
            errs().write_escaped(F.getName()) << '\n');
      // Only consider definition of SYCL kernels
      if (!F.isDeclaration() && sycl::isKernel(F))
          serializeKernelArguments(F);
    }

    // The module probably changed
    return true;
  }
};

}

char SYCLSerializeArguments::ID = 0;
static RegisterPass<SYCLSerializeArguments> X {
  "SYCL-serialize-arguments",
  "pass to serialize arguments of a SYCL kernel"
 };
