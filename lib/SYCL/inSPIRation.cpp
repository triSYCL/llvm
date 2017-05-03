//===- inSPIRation.cpp                           ---------------===//
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
#include "llvm/IR/Constants.h"
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
struct inSPIRation : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  inSPIRation() : ModulePass(ID) {}


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

    // This is a SPIR kernel
    F.setCallingConv(CallingConv::SPIR_KERNEL);

    // A SPIR kernel has no personality
    F.setPersonalityFn(nullptr);

    /* Add kernel metadata inSPIRed from GenOpenCLArgMetadata() in
       /tools/clang/lib/CodeGen/CodeGenFunction.cpp */

    auto &Ctx = F.getContext();
    auto Int32Ty = llvm::Type::getInt32Ty(Ctx);

    // MDNode for the kernel argument address space qualifiers
    SmallVector<llvm::Metadata *, 8> AddressSpaceQuals;

    for (auto &A : F.args()) {
      if (auto PTy = dyn_cast<PointerType>(A.getType())) {
        // Add numeric value of the address space as address qualifier
        AddressSpaceQuals.push_back(
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(Int32Ty,
                                       PTy->getAddressSpace())));
      }
      else
        // Otherwise use default address space
        AddressSpaceQuals.push_back(
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(Int32Ty, 0)));
    }
    //  Add the SPIR metadata describing the address space of each argument
    F.setMetadata("kernel_arg_addr_space",
                  llvm::MDNode::get(Ctx, AddressSpaceQuals));
  }


/// Add metadata for the SPIR 2.0 version
void setSPIRVersion(Module &M) {
  /* Get inSPIRation from SPIRTargetCodeGenInfo::emitTargetMD in
     tools/clang/lib/CodeGen/TargetInfo.cpp */
  auto &Ctx = M.getContext();
  auto Int32Ty = llvm::Type::getInt32Ty(Ctx);
  // SPIR v2.0 s2.12 - The SPIR version used by the module is stored in the
  // opencl.spir.version named metadata.
  llvm::Metadata *SPIRVerElts[] = {
    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 2)),
    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 0))
  };
  M.getOrInsertNamedMetadata("opencl.spir.version")
    ->addOperand(llvm::MDNode::get(Ctx, SPIRVerElts));
}


/// Add metadata for the OpenCL 1.2 version
void setOpenCLVersion(Module &M) {
  /* Get inSPIRation from SPIRTargetCodeGenInfo::emitTargetMD in
     tools/clang/lib/CodeGen/TargetInfo.cpp */
  auto &Ctx = M.getContext();
  auto Int32Ty = llvm::Type::getInt32Ty(Ctx);
  // SPIR v2.0 s2.13 - The OpenCL version used by the module is stored in the
  // opencl.ocl.version named metadata node.
  llvm::Metadata *OCLVerElts[] = {
    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 1)),
    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 2))
  };
  llvm::NamedMDNode *OCLVerMD =
      M.getOrInsertNamedMetadata("opencl.ocl.version");
  OCLVerMD->addOperand(llvm::MDNode::get(Ctx, OCLVerElts));
}


  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    for (auto &F : M.functions()) {
      // Only consider definition of SYCL kernels
      // \todo Put SPIR calling convention on declarations too
      if (!F.isDeclaration() && sycl::isKernel(F))
          kernelSPIRify(F);
    }

    setSPIRVersion(M);

    setOpenCLVersion(M);

    // The module probably changed
    return true;
  }
};

}

char inSPIRation::ID = 0;
static RegisterPass<inSPIRation> X {
  "inSPIRation",
  "pass to make functions and kernels SPIR compatible"
 };
