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
// representation as described in "The SPIR Specification Version 2.0 -
// Provisional" from Khronos Group.
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/CallSite.h"
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
STATISTIC(SYCLFuncCalledInKernelFound, "Number of functions directly or indirectly called by SYCL kernel functions");

cl::opt<bool> ReqdWorkGroupSizeOne("reqd-workgroup-size-1", cl::desc("set reqd_work_group_size to be 1-1-1"));

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Transform the SYCL kernel functions into SPIR-compatible kernels
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


  /// Construct an equivalent SPIR typename compatible with OpenCL kernel
  /// calling conventions
  ///
  /// \todo Use a less hackish way to prettyprint the right types.
  ///
  /// \todo Implement more types from section "2.1 Supported Data Types" of "The
  /// SPIR Specification Version 2.0 - Provisional" document.
  std::string buildSPIRType(const Argument &A) {
    // First get the LLVM type as a string
    std::string TypeName;
    raw_string_ostream SO { TypeName };
    SO << *A.getType();
    SO.flush();
    // A list of rewriting as std::regex/format pairs to be used by
    // std::regex_replace
    const char * Type_Transforms[][2] = {
      { "i8", "char" },
      { "i16", "short" },
      { "i32", "int" },
      { "i64", "long" },
      // Has to appear after "i16" to be deterministic:
      { "i1", "bool" },
      // Suppress the address space information
      { "addrspace\\(.\\)", "" }
    };
    // Apply the type rewriting recipe
    for (auto &Transform : Type_Transforms)
      TypeName = std::regex_replace(TypeName,
                                    std::regex { Transform[0] },
                                    Transform[1]);
    return TypeName;
  }


  /// Transform a function into a SPIR-compatible kernel
  void kernelSPIRify(Function &F) {
    ++SYCLKernelProcessed;

    // This is a SPIR kernel
    F.setCallingConv(CallingConv::SPIR_KERNEL);

    // A SPIR kernel has no personality
    F.setPersonalityFn(nullptr);

    /* Add kernel metadata inSPIRed from GenOpenCLArgMetadata() in
       tools/clang/lib/CodeGen/CodeGenFunction.cpp */

    auto &Ctx = F.getContext();
    auto Int32Ty = llvm::Type::getInt32Ty(Ctx);

    // MDNode for the kernel argument address space qualifiers
    SmallVector<llvm::Metadata *, 8> AddressSpaceQuals;
    // MDNode for the kernel argument types
    SmallVector<llvm::Metadata *, 8> Types;
    // MDNode for the kernel argument type qualifiers
    SmallVector<llvm::Metadata *, 8> TypeQuals;
    // MDNode for the kernel argument access qualifiers
    SmallVector<llvm::Metadata *, 8> AccessQuals;
    // MDNode for the kernel required work group size
    SmallVector<llvm::Metadata *, 8> ReqdWorkGroupSize;

    for (auto &A : F.args()) {
      std::string TypeName;
      raw_string_ostream SO { TypeName };
      SO << *A.getType();
      SO.flush();
      std::regex RE_i32 { "i32" };
      DEBUG(dbgs() << "Type name " << TypeName
            << '\n' << std::regex_replace(TypeName, RE_i32, "int")
            << '\n' << buildSPIRType(A)<< '\n';
            A.getType()->dump(););
      Types.push_back(llvm::MDString::get(Ctx, buildSPIRType(A)));

      std::string TypeQual;
      auto buildTypeQualString = [&] (bool Present, const char *SPIRName) {
        if (Present) {
          if (TypeQual.empty())
            TypeQual = SPIRName;
          else
            TypeQual += std::string { " " } + SPIRName;
        }
      };
      buildTypeQualString(A.onlyReadsMemory(), "const");
      buildTypeQualString(A.hasNoAliasAttr(), "restrict");
      // \todo Deal with volatile
      // \todo Deal with pipes
      TypeQuals.push_back(llvm::MDString::get(Ctx, TypeQual));

      // \todo Deal with kernel arg access qual
      AccessQuals.push_back(llvm::MDString::get(Ctx, "read_write"));

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

    for (int i = 0; i < 3; ++i) {
      ReqdWorkGroupSize.push_back(
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(Int32Ty, 1)));
    }

    // Add the SPIR metadata describing the address space of each argument
    F.setMetadata("kernel_arg_addr_space",
                  llvm::MDNode::get(Ctx, AddressSpaceQuals));

    // Add the SPIR metadata describing the type of each argument
    F.setMetadata("kernel_arg_type", llvm::MDNode::get(Ctx, Types));

    // For now, just repeat "kernel_arg_type" as "kernel_arg_base_type" because
    // we do not have the type alias information
    F.setMetadata("kernel_arg_base_type", llvm::MDNode::get(Ctx, Types));

    // Add the SPIR metadata describing the type qualifier of each argument
    F.setMetadata("kernel_arg_type_qual", llvm::MDNode::get(Ctx, TypeQuals));

    // Add the SPIR metadata describing the access qualifier of each argument
    F.setMetadata("kernel_arg_access_qual",
                  llvm::MDNode::get(Ctx, AccessQuals));

    if (ReqdWorkGroupSizeOne) {
      // Add the SPIR metadata for required work group size
      F.setMetadata("reqd_work_group_size",
                    llvm::MDNode::get(Ctx, ReqdWorkGroupSize));
    }
  }


  /// Replace the function called in kernel to SPIR calling convention
  void kernelCallFuncSPIRify(Function &F) {
    SYCLFuncCalledInKernelFound++;
    // This is a SPIR function
    DEBUG(dbgs() << F.getName() << "is a SPIR function.\n");
    F.setCallingConv(CallingConv::SPIR_FUNC);
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


  /// Set the output Triple to SPIR
  void setSPIRTriple(Module &M) {
    M.setTargetTriple("spir64");
  }


  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    // funcCount is for naming new name for each function called in kernel
    int funcCount = 0;

    for (auto &F : M.functions()) {
      // Only consider definition of SYCL kernels
      // \todo Put SPIR calling convention on declarations too
      if (!F.isDeclaration()) {
        if (sycl::isKernel(F)) {
          kernelSPIRify(F);

          // Rename basic block name
          int count = 0;
          for (auto &B : F)
            B.setName("label_" + Twine{count++});
        } else if (!F.isIntrinsic()) {
          // After kernels code selection, there are only two kinds of functions
          // left: funcions called by kernels or LLVM intrinsic functions.
          // For functions called in SYCL kernels, put SPIR calling convention.
          kernelCallFuncSPIRify(F);

          // Modify the name of funcions called by SYCL kernel since function
          // names with $ sign would choke Xilinx xocc.
          // And in Xilinx xocc, there are passes splitting a function to new
          // functions. These new function names will come from some of the
          // basic block names in the original function.
          // So function and basic block names need to be modified to avoid
          // containing $ sign

          // Rename function name
          F.setName("sycl_func_" + Twine{funcCount++});

          // Rename basic block name
          int count = 0;
          for (auto &B : F)
            B.setName("label_" + Twine{count++});
        }
      }
    }

    setSPIRVersion(M);

    setOpenCLVersion(M);

    setSPIRTriple(M);

    //setSPIRLayout(M);

    // The module probably changed
    return true;
  }
};

}

char inSPIRation::ID = 0;
static RegisterPass<inSPIRation> X {
  "inSPIRation",
  "pass to make functions and kernels SPIR-compatible"
 };
