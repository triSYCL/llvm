//===- remove_global_empty_cdtors.cpp                       ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Remove empty list of global constructors or destructors (RELGCD).
//
// The global dead-code elimination remove the useless constructors and
// destructors from the code and also from "@llvm.global_ctors" and
// "@llvm.global_dtors".
//
// But when these arrays are empty they remain in the code and that chokes some
// SPIR consumers such as Xilinx xocc, as it is not legal SPIR:
// \code
// @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
// \endcode
// So this pass removes these empty "@llvm.global_ctors" and
// "@llvm.global_dtors"
//
// \todo Move this into the official Global Dead Code Elimination pass.
// ===---------------------------------------------------------------------===//

#include <cstddef>

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/SYCL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
/// option -debug or -debug-only=SYCL
#define DEBUG_TYPE "SYCL"


/// Displayed with -stats
STATISTIC(RemovedEmptyGlobalConstructors,
          "Number of global empty constructor list @llvm.global_ctors removed");
STATISTIC(RemovedEmptyGlobalDestructors,
          "Number of global empty destructor list @llvm.global_dtors removed");

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Replace a SYCL kernel code by a function serializing its arguments
struct RELGCD : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  RELGCD() : ModulePass(ID) {}


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


/// Remove a global variable of the given name if it is pointer to an empty
/// array
bool removeEmptyGlobalArray(Module &M,
                            const char *GlobalVariableName,
                            llvm::Statistic &S) {
  auto GL = M.getGlobalVariable(GlobalVariableName);
  if (GL) {
    DEBUG(errs() << "Found " << GlobalVariableName << "\n\n");
    GL->getType()->dump();
    if (auto PT = dyn_cast<PointerType>(GL->getType()))
      // If it is an array, try to get the pointee array
      if (auto AT = dyn_cast<ArrayType>(PT->getElementType()))
        if (AT->getNumElements() == 0) {
          // If the pointed array is of size 0, remove the useless global
          // variable
          GL->eraseFromParent();
          // Update the statistic for this kind of variable removed
          ++S;
          // The code has changed
          return true;
        }
  }
  // Nothing done
  return false;
}


  /// Remove the global variable to the empty arrays of contructors or
  /// destructors
  bool runOnModule(Module &M) override {
    // This is really a non short-cut "boolean or" here
    return removeEmptyGlobalArray(M,
                                  "llvm.global_ctors",
                                  RemovedEmptyGlobalConstructors)
      | removeEmptyGlobalArray(M,
                               "llvm.global_dtors",
                               RemovedEmptyGlobalDestructors);
  }

};

}

char RELGCD::ID = 0;
static RegisterPass<RELGCD> X {
  "RELGCD",
  "pass to make functions and kernels SPIR compatible"
 };
