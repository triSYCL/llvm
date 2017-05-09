//===- SYCLSerializeArgumentsInside.cpp                        ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Replace a SYCL kernel code by a function serializing its arguments.
//
// The kernel body is replaced by the serialization code from inside.
// ===---------------------------------------------------------------------===//

#include <cstddef>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
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
struct SYCLSerializeArgumentsInside : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  /// The mangled name of the serialization function to use.
  ///
  /// Note that it has to be defined in some include files so this pass can use
  /// it.
  ///
  /// The function is defined
  /// in triSYCL/include/CL/sycl/device_runtime.hpp
  ///
  /// TRISYCL_WEAK_ATTRIB_PREFIX void TRISYCL_WEAK_ATTRIB_SUFFIX
  /// serialize_arg(detail::task &task,
  ///               std::size_t index,
  ///               void *arg,
  ///               std::size_t arg_size)
  static auto constexpr SerializationFunctionName =
    "_ZN2cl4sycl3drt13serialize_argERNS0_6detail4taskEmPvm";


  /// The mangled name of the kernel launching function to use.
  ///
  /// Note that it has to be defined in some include files so this pass can use
  /// it.
  ///
  /// The function is defined
  /// in triSYCL/include/CL/sycl/device_runtime.hpp
  ///
  /// TRISYCL_WEAK_ATTRIB_PREFIX void TRISYCL_WEAK_ATTRIB_SUFFIX
  /// launch_kernel(detail::task &task,
  ///               const char *kernel_name)
  static auto constexpr KernelLaunchingFunctionName =
    "_ZN2cl4sycl3drt13launch_kernelERNS0_6detail4taskEPKc";


  SYCLSerializeArgumentsInside() : ModulePass(ID) {}


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

    // An iterator pointing to the first function argument
    auto A = F.arg_begin();
    // The first argument is the cl::sycl::detail::task address
    auto &Task = *A++;

    // The index used to number the arguments in the serialization
    std::size_t IndexNumber = 0;
    // Deal with the remaining arguments
    for (; A != F.arg_end(); ++A) {
      DEBUG(dbgs() << "Serializing '" << A->getName() << "'.\n");
      DEBUG(dbgs() << "Size '" << DL.getTypeAllocSize(A->getType()) << "'.\n");
      // An IR version of the index number
      auto Index = Builder.getInt64(IndexNumber);

      // \todo Refactor/fuse the then/else part
      if (auto PTy = dyn_cast<PointerType>(A->getType())) {
        DEBUG(dbgs() << " pointer to\n");
        DEBUG(PTy->getElementType()->dump());
        // The pointer argument casted to a void *
        auto Arg =
          Builder.CreatePointerCast(&*A, Type::getInt8PtrTy(F.getContext()));
        // The size of the pointee type
        auto ArgSize = DL.getTypeAllocSize(PTy->getElementType());
        // Insert the call to the serialization function with the 3 required
        // arguments
        Value * Args[] { &Task, Index, Arg, Builder.getInt64(ArgSize) };
        // \todo add an initializer list to makeArrayRef
        Builder.CreateCall(SF, makeArrayRef(Args));
      }
      else {
        // Create an intermediate memory place to pass the value by address
        auto Alloca = Builder.CreateAlloca(A->getType());
        Builder.CreateStore(&*A, Alloca);
        auto Arg =
          Builder.CreatePointerCast(Alloca,
                                    Type::getInt8PtrTy(F.getContext()));
        // The size of the argument
        auto ArgSize = DL.getTypeAllocSize(A->getType());
        // Insert the call to the serialization function with the 3 required
        // arguments
        Value * Args[] { &Task, Index, Arg, Builder.getInt64(ArgSize) };
        // \todo add an initializer list to makeArrayRef
        Builder.CreateCall(SF, makeArrayRef(Args));
      }
      ++IndexNumber;
    }

    // Get the predefined kernel launching function to use
    auto KLF = F.getParent()->getValueSymbolTable()
      .lookup(KernelLaunchingFunctionName);
    assert(KLF && "Kernel launching function not found");

    // Create a global string variable with the name of the kernel itself
    // and return an char * on it
    auto Name = Builder.CreateGlobalStringPtr(F.getName());

    // Add the launching of the kernel
    Value * Args[] { &Task, Name };
    // \todo add an initializer list to makeArrayRef
    Builder.CreateCall(KLF, makeArrayRef(Args));

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

char SYCLSerializeArgumentsInside::ID = 0;
static RegisterPass<SYCLSerializeArgumentsInside> X {
  "SYCL-serialize-arguments-inside",
  "pass to serialize arguments of a SYCL kernel"
 };
