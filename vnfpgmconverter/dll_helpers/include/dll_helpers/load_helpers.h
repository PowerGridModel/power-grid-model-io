#pragma once
#ifndef DLL_HELPERS_LOAD_HELPERS_HPP
#define DLL_HELPERS_LOAD_HELPERS_HPP

#if _WIN32
#include "Windows.h"
#include "tchar.h"
#else
#include "dlfcn.h"
#endif

#if _WIN32
#define PGM_VNF_CONVERTER_CALL __stdcall
#define PGM_VNF_CONVERTER_HANDLE HMODULE
#define PGM_VNF_CONVERTER_SYMBOL FARPROC
#define PGM_VNF_CONVERTER_DLL_PREFIX ""
#define PGM_VNF_CONVERTER_DLL_SUFFIX ".dll"
#else
#define PGM_VNF_CONVERTER_CALL
#define PGM_VNF_CONVERTER_HANDLE void*
#define PGM_VNF_CONVERTER_SYMBOL void*
#define PGM_VNF_CONVERTER_DLL_PREFIX "lib"
#define PGM_VNF_CONVERTER_DLL_SUFFIX ".so"
#endif

inline PGM_VNF_CONVERTER_HANDLE load_library(const char* library_name) {
#ifdef _WIN32
    return LoadLibraryA(library_name);
#else
    return dlopen(library_name, RTLD_LAZY);
#endif
}

inline int unload_library(PGM_VNF_CONVERTER_HANDLE handle) {
    if (!handle) {
        return -1;
    }

#ifdef _WIN32
    return FreeLibrary(handle);
#else
    return dlclose(handle);
#endif
}

inline PGM_VNF_CONVERTER_SYMBOL get_function(PGM_VNF_CONVERTER_HANDLE handle, const char* function_name) {
#ifdef _WIN32
    return (PGM_VNF_CONVERTER_SYMBOL)GetProcAddress(handle, function_name);
#else
    return (PGM_VNF_CONVERTER_SYMBOL)dlsym(handle, function_name);
#endif
}

#endif
