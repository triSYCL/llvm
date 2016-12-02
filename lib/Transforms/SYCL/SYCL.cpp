//===- SYCL.cpp                                             ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect SYCL kernels
//
//===----------------------------------------------------------------------===//

#include <string>

#include "llvm/ADT/Statistic.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CtorUtils.h"

using namespace llvm;

// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
// option -debug or -debug-only=SYCL
#define DEBUG_TYPE "SYCL"


StringRef SYCLKernelPrefix { "void cl::sycl::detail::instantiate_kernel<" };

/// Test if a function is a SYCL kernel
bool isSYCLKernel(const Function &F) {
  bool KernelFound = false;
  // Demangle C++ name for human beings
  int Status;
  char *Demangled = itaniumDemangle(F.getName().str().c_str(),
                                    nullptr,
                                    nullptr,
                                    &Status);
  if (Demangled) {
    DEBUG(errs() << " Demangled: " << Demangled);
    if (StringRef { Demangled }.startswith(SYCLKernelPrefix)) {
      DEBUG(errs() << " \n\n\tKernel found!\n\n");
      KernelFound = true;
    }
  }
  free(Demangled);
  return KernelFound;
}


// Displayed with -stats
STATISTIC(SYCLCounter, "Processed functions");

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

// Detect SYCL kernel use
struct SYCL : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  // The prefix of a SYCL kernel name in a module
  static StringRef KernelPrefix;


  SYCL() : ModulePass(ID) {}


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


  // Mark kernels as external so a GlobalDCE pass will keep them
  void handleKernel(Function &F) {
    F.setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  }


  // Mark non kernels with internal so a GlobalDCE pass may discard
  // them if they are not used
  void handleNonKernel(Function &F) {
    DEBUG(errs() << "\tmark function with InternalLinkage: ";
          errs().write_escaped(F.getName()) << '\n');
    F.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  }


  /// Visit all the basic-blocks
  bool runOnModule(Module &M) override {
    for (auto &F : M.functions()) {
      DEBUG(errs() << "Function: ";
            errs().write_escaped(F.getName()) << '\n');
      // Only consider definition of functions
      if (!F.isDeclaration()) {
        if (isSYCLKernel(F))
          handleKernel(F);
        else
          handleNonKernel(F);
      }
    }

    for (auto &G : M.globals()) {
      DEBUG(errs() << "Global: " << G.getName() << '\n');
      // Skip intrinsic variable for now.
      // Factorize out Function::isIntrinsic to something higher?
      if (!G.isDeclaration()
          && !G.getName().startswith("llvm."))
        G.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }

    // Make the global alias internal too otherwise the GlobalDCE will think
    // these objects are useful
    for (GlobalAlias &GA : M.aliases()) {
      GA.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }

    // Remove also the global destructors.
    // For now just consider that a kernel cannot have some program-scope
    // constructors
    optimizeGlobalCtorsList(M, [] (Function *) { return true; });

    // The module probably changed
    return true;
  }
};

}

char SYCL::ID = 0;
static RegisterPass<SYCL> X { "SYCL-filter-kernel",
                              "SYCL kernel detection pass" };
