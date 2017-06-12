//===- SYCLKernelFilter.cpp                               ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect and mark SYCL kernels with external linkage.  Everything else is
// marked with internal linkage, so the GlobalDCE pass can be used later to
// keep only the kernel code and the transitive closure of the dependencies
// ===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
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
STATISTIC(SYCLKernelFound, "Number of SYCL kernel functions");
STATISTIC(SYCLNonKernelFound, "Number of non SYCL kernel functions");

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Detect and mark SYCL kernels with external linkage
///
/// Everything else is marked with internal linkage, so the GlobalDCE pass
/// can be used later to keep only the kernel code and the transitive closure
/// of the dependencies.
///
/// Based on an idea from Mehdi Amini
struct SYCLKernelFilter : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  SYCLKernelFilter() : ModulePass(ID) {}


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


  /// Mark kernels as external so the GlobalDCE pass will keep them and rename
  /// them to a simpler SPIR-compatible name, just in case...
  void handleKernel(Function &F) {
    F.setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
    // Rename the kernel to a simpler unique name
    F.setName(sycl::registerSYCLKernelAndGetShortName(F.getName()));
    SYCLKernelFound++;
  }


  /// Mark non kernels with internal linkage so the GlobalDCE pass may discard
  /// them if they are not used
  void handleNonKernel(Function &F) {
    DEBUG(errs() << "\tmark function with InternalLinkage: ";
          errs().write_escaped(F.getName()) << '\n');
    F.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    SYCLNonKernelFound++;
  }


  /// Visit all the module content
  bool runOnModule(Module &M) override {
    for (auto &F : M.functions()) {
      DEBUG(errs() << "Function: ";
            errs().write_escaped(F.getName()) << '\n');
      // Only consider definition of functions
      if (!F.isDeclaration()) {
        if (sycl::isKernel(F))
          handleKernel(F);
        else
          handleNonKernel(F);
      }
    }

    // The global variables may keep references to some functions, so mark them
    // as internal too
    for (auto &G : M.globals()) {
      DEBUG(errs() << "Global: " << G.getName() << '\n');
      // Skip intrinsic variable for now.
      // \todo Factorize out Function::isIntrinsic to something higher?
      if (!G.isDeclaration()
          && !G.getName().startswith("llvm."))
        G.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }

    // Make the global aliases internal too, otherwise the GlobalDCE will think
    // the aliased objects are useful
    for (GlobalAlias &GA : M.aliases()) {
      GA.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }

    // Remove also the global destructors.  For now, just consider that
    // a kernel cannot have program-scope (in the sense of OpenCL) constructors
    optimizeGlobalCtorsList(M, [] (Function *) { return true; });

    // The module probably changed
    return true;
  }
};

}

char SYCLKernelFilter::ID = 0;
static RegisterPass<SYCLKernelFilter> X { "SYCL-kernel-filter",
                                          "SYCL kernel filtering pass" };
