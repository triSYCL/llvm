# If we don't need RTTI or EH, there's no reason to export anything
# from the hello plugin.
if( NOT LLVM_REQUIRES_RTTI )
  if( NOT LLVM_REQUIRES_EH )
    set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/SYCL.exports)
  endif()
endif()

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_loadable_module( SYCL
  ItaniumDemangle.cpp
  SYCL-annotation.cpp
  SYCLArgsFlattening.cpp
  SYCLKernel.cpp
  SYCLKernelFilter.cpp
  SYCLSerializeArguments.cpp
  SYCLSerializeArgumentsInside.cpp
  inSPIRation.cpp
  remove_global_empty_cdtors.cpp

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )
