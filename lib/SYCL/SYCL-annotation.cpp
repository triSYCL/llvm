//===- SYCL-annotation.cpp                                  ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect SYCL kernels based on annotation
//
// This expect in the C++ runtime something like:
// \code
// __attribute__((annotate("__triSYCL_kernel")))
// \endcode
// to mark kernels
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

using namespace llvm;

// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
// option -debug or -debug-only=SYCL
#define DEBUG_TYPE "SYCL"


// Displayed with -stats
STATISTIC(SYCL_functions, "Processed functions");

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

// Detect SYCL kernel annotation use
struct SYCL_annotation : public BasicBlockPass {

  static char ID; // Pass identification, replacement for typeid

  // The annotation string used to mark triSYCL kernels.
  //
  // It is actually quite uneasy to build strings ended with \0 even in C++...
  const std::string SYCL_kernel_mark =
    std::string { "__triSYCL_kernel" } + '\0';

  SYCL_annotation() : BasicBlockPass(ID) {}

  // Pass initialization for each function
  bool doInitialization(Function &F) override {
    SYCL_functions++;
    // Do not change the code
    return false;
  }


  void DealWithSYCLAnnotation(IntrinsicInst *II) {
    DEBUG(errs() << "Found __triSYCL_kernel marker in module ");
    DEBUG(errs().write_escaped(II->getModule()->getName()));
    DEBUG(errs() << " and function ");
    DEBUG(errs().write_escaped(II->getFunction()->getName()) << '\n');
    DEBUG(
        // Demangle C++ name for human beings
        int Status;
        char *Demangled =
          itaniumDemangle(II->getFunction()->getName().str().c_str(),
                          nullptr,
                          nullptr,
                          &Status);
        if (Demangled)
          errs() << " Demangled: " << Demangled << '\n';
        free(Demangled);
          );
    // Chase the kernel functor
    auto F = II->getOperand(0);
    // This is typically a cast instruction like
    // %f4 = bitcast %class.anon.173* %f to i8*
    DEBUG(errs() << "Annotated functor: ");
    DEBUG(F->dump());
    if (const auto *FuncPtr = dyn_cast<Instruction>(F)) {
      if (const auto *BC = dyn_cast<BitCastInst>(FuncPtr)) {
        // Extract the functor type from the source pointer type
        auto ST = BC->getSrcTy();
        if (auto PT = dyn_cast<PointerType>(ST)) {
          auto T = PT->getElementType();
          DEBUG(errs() << "Functor kernel type capturing the accessors: ");
          DEBUG(T->dump());
        }
      }
    }
  }


  /// Visit all the basic-blocks
  bool runOnBasicBlock(BasicBlock &BB) override {
    // Look for a var_annotation that flags a SYCL kernel use
    //
    // A typical use case is

    // \code{llvm}
    // @.str.19 = private unnamed_addr constant [17 x i8] c"__triSYCL_kernel\00", section "llvm.metadata"
    // @.str.20 = private unnamed_addr constant [78 x i8] c"/home/keryell/Xilinx/Projects/OpenCL/SYCL/triSYCL/include/CL/sycl/handler.hpp\00", section "llvm.metadata"
    // %class.anon.173 = type { %"class.cl::sycl::accessor.162" }
    //
    // define internal void @"_ZN2cl4sycl7handler12parallel_forIZZ4mainENK3$_1clERS1_E7nothingZZ4mainENKS3_clES4_EUliE_EEvNS0_5rangeILm1EEET0_"(%"class.cl::sycl::handler"* %this, i64 %global_size.coerce, %class.anon.173* %f) #0 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !9914 {
    //
    // %f4 = bitcast %class.anon.173* %f to i8*
    //   call void @llvm.var.annotation(i8* %f4, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.19, i32 0, i32 0), i8* getelementptr inbounds ([78 x i8], [78 x i8]* @.str.20, i32 0, i32 0), i32 217)
    // \endcode
    //
    // The following code is basically infered from llvm/lib/IR/AsmWriter.cpp
    for (auto &I : BB)
      if (auto II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::var_annotation) {
          DEBUG(II->dump());
          // Check this is a triSYCL kernel markup.
          // Operand 1 should be GEP ConstantExpr to the SYCL markup string
          auto CV = II->getOperand(1);
          if (auto CE = dyn_cast<ConstantExpr>(CV))
            if (isa<GEPOperator>(CE)) {
              auto AnnSymbol = CE->getOperand(0);
              if (auto GV = dyn_cast<GlobalVariable>(AnnSymbol))
                if (GV->hasInitializer())
                  if (auto C = dyn_cast<Constant>(GV->getInitializer()))
                    if (auto CA = dyn_cast<ConstantDataArray>(C))
                      if (CA->isString()
                          && (CA->getAsString() == SYCL_kernel_mark))
                        DealWithSYCLAnnotation(II);
            }
        }
    // Do not change the code
    return false;
  }
};

}

char SYCL_annotation::ID = 0;
static RegisterPass<SYCL_annotation>
X { "SYCL-annotation", "SYCL kernel annotation detection pass" };
