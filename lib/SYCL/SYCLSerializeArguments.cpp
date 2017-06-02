//===- SYCLSerializeArguments.cpp                           ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Replace a SYCL kernel code by a function serializing its arguments and
// calling the kernel.
//
// Now the kernel argument serialization & kernel call are done by the outside.
//
// Basically we look for the functions containing a call to a kernel and we
// transform
// \code
//   tail call void @_ZN2cl4sycl6detail22set_kernel_task_markerERNS1_4taskE(%"struct.cl::sycl::detail::task"* nonnull dereferenceable(240) %24)
//   %25 = getelementptr inbounds %class.anon.439, %class.anon.439* %1, i64 0, i32 1
//   tail call fastcc void @"_ZN2cl4sycl6detail18instantiate_kernelIZZ9test_mainiPPcENK3$_1clERNS0_7handlerEE3addZZ9test_mainiS4_ENKS5_clES7_EUlvE_EEvT0_"(%class.anon.433* byval nonnull align 8 %25)
// \endcode
// into
// \code
//   %26 = addrspacecast i32 addrspace(1)* %.idx.val to i8*
//   call void @_ZN2cl4sycl3drt13serialize_argERNS0_6detail4taskEmPvm(%"struct.cl::sycl::detail::task"* %24, i64 0, i8* %26, i64 4)
//   %27 = addrspacecast i32 addrspace(1)* %.idx1.val to i8*
//   call void @_ZN2cl4sycl3drt13serialize_argERNS0_6detail4taskEmPvm(%"struct.cl::sycl::detail::task"* %24, i64 1, i8* %27, i64 4)
//   call void @_ZN2cl4sycl3drt13launch_kernelERNS0_6detail4taskEPKc(%"struct.cl::sycl::detail::task"* %24, i8* getelementptr inbounds ([173 x i8], [173 x i8]* @0, i32 0, i32 0))
// \endcode
// by including also the effect of the SYCLArgsFlattening pass.
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <functional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
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


  /// The mangled name of the function marking the task to be used to launch the
  /// kernel.
  ///
  /// Note that it has to be defined in some include files so this pass can
  /// find it.
  ///
  /// The function is defined
  /// in triSYCL/include/CL/sycl/detail/instantiate_kernel.hpp
  ///
  /// extern void set_kernel_task_marker(detail::task &task)
  static auto constexpr SetKernelTaskFunctionName =
    "_ZN2cl4sycl6detail22set_kernel_task_markerERNS1_4taskE";


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
  /// set_kernel(detail::task &task,
  ///            const char *kernel_name)
  static auto constexpr SetKernelFunctionName =
    "_ZN2cl4sycl3drt10set_kernelERNS0_6detail4taskEPKc";


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


  /// Replace the kernel call instructions by the serialization of its arguments
  void serializeKernelArguments(Function &F, Instruction &KernelCall) {
    ++SYCLKernelProcessed;

    // Find the call to \c set_kernel_task() in before the kernel call
    Instruction *SetKernelTaskInstruction = nullptr;
    CallSite SetKernelTaskCS;
    for (auto I = (&KernelCall)->getPrevNode(); I; I = I->getPrevNode())
      if (auto CS = CallSite { I }) {
        auto CF = CS.getCalledFunction();
        // Is this a call to the right function?
        if (CF->hasName() && CF->getName().str() == SetKernelTaskFunctionName) {
          SetKernelTaskInstruction = I;
          SetKernelTaskCS = CS;
          break;
        }
      }
    assert(SetKernelTaskInstruction
           && "There should a call to set_kernel_task_marker()"
           "before the call to the kernel");

    // Need the data layout of the target to measure object size
    auto DL = F.getParent()->getDataLayout();

    // Get the predefined serialization function to use
    auto SF = F.getParent()->getValueSymbolTable()
      .lookup(SerializationFunctionName);
    assert(SF && "Serialization function not found");

    /* Get the cl::sycl::detail::task address which is passed as the argument of
       the marking function */
    auto &Task = *SetKernelTaskCS.getArgument(0);

    // Use an IRBuilder to ease IR creation in the basic block
    auto BB = KernelCall.getParent();
    IRBuilder<> Builder { BB };
    // Insert the future new instructions before the current kernel call
    Builder.SetInsertPoint(&KernelCall);

    // Get the predefined kernel setting function to use
    auto SKF = F.getParent()->getValueSymbolTable()
      .lookup(SetKernelFunctionName);
    assert(SKF && "Kernel setting function not found");

    // Create a global string variable with the name of the kernel itself
    // and return a char * on it
    auto Name = Builder.CreateGlobalStringPtr(F.getName());

    // Add the setting of the kernel
    Value * Args[] { &Task, Name };
    // \todo add an initializer list to makeArrayRef
    Builder.CreateCall(SKF, makeArrayRef(Args));

    CallSite KernelCallSite { &KernelCall };
    // The index used to number the arguments in the serialization
    std::size_t IndexNumber = 0;
    // Iterate on the kernel call arguments
    for (auto &A : KernelCallSite.args()) {
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

    // Now remove the marking function and the initial kernel call
    SetKernelTaskInstruction->eraseFromParent();
    KernelCall.eraseFromParent();
  }


  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    // First find the kernel calling side independently to avoid rewriting the
    // world we iterate on
    SmallVector<std::pair<std::reference_wrapper<Function>,
                          std::reference_wrapper<Instruction>>,
                8> KernelCallSites;
    for (auto &F : M.functions()) {
      // Look for calls by this function
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto CS = CallSite { &I })
            // If we call a kernel, it is a kernel call site
            if (auto CF = CS.getCalledFunction())
              if (sycl::isKernel(*CF))
                KernelCallSites.emplace_back(F, I);
    }
    // Then serialize the calls to the detected kernels
    for (auto &KernelCall : KernelCallSites)
      serializeKernelArguments(KernelCall.first, KernelCall.second);

    // The module changed if there were some kernels
    return !KernelCallSites.empty();
  }

};

}

char SYCLSerializeArguments::ID = 0;
static RegisterPass<SYCLSerializeArguments> X {
  "SYCL-serialize-arguments",
  "pass to serialize arguments of a SYCL kernel"
 };
