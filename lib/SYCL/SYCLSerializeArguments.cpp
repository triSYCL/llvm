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

#include <cstddef>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"
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
struct SYCLSerializeArguments : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  /// The mangled name of the serialization function to use.
  ///
  /// Note that it has to be defined in some include files so this pass can use
  /// it.
  static auto constexpr SerializationFunctionName =
    "_ZN2cl4sycl3drt13serialize_argEmPvm";


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

    // Need the data layout of the target to measure object size
    auto DL = F.getParent()->getDataLayout();

    // Get the predefined serialization function to use
    auto SF = F.getParent()->getValueSymbolTable()
      .lookup(SerializationFunctionName);
    assert(SF && "Serialization function not found");

    // The index used to number the arguments in the serialization
    std::size_t IndexNumber = 0;
    for (Argument &A : F.args()) {
      DEBUG(dbgs() << "Serializing '" << A.getName() << "'.\n");
      DEBUG(dbgs() << "Size '" << DL.getTypeAllocSize(A.getType()) << "'.\n");

      if (auto PTy = dyn_cast<PointerType>(A.getType())) {
        DEBUG(dbgs() << " pointer to\n");
        DEBUG(PTy->getElementType()->dump());
        auto Index = Builder.getInt64(IndexNumber);
        // The pointer argument casted to a void *
        auto Arg =
          Builder.CreatePointerCast(&A, Type::getInt8PtrTy(F.getContext()));
        // The size of the pointee type
        auto ArgSize = DL.getTypeAllocSize(PTy->getElementType());
        // Insert the call to the serialization function with the 3 required
        // arguments
        Value * Args[] { Index, Arg, Builder.getInt64(ArgSize) };
        // \todo add an initializer list to makeArrayRef
        Builder.CreateCall(SF, makeArrayRef(Args));
      }
      ++IndexNumber;
    }

    // Add a "ret void" as the function terminator.
    // Assume the return type of a kernel is void.
    Builder.CreateRetVoid();
  }


  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    for (auto &F : M.functions()) {
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
