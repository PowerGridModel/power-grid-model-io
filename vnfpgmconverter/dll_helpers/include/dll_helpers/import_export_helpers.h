// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once
#ifndef DLL_HELPERS_IMPORT_EXPORT_HELPERS_H
#define DLL_HELPERS_IMPORT_EXPORT_HELPERS_H
#endif

// Generic helper definitions for shared library support
#if defined _WIN32
#define PGM_VNF_CONVERTER_HELPER_DLL_IMPORT __declspec(dllimport)
#define PGM_VNF_CONVERTER_HELPER_DLL_EXPORT __declspec(dllexport)
#define PGM_VNF_CONVERTER_HELPER_DLL_LOCAL
#else
#if __GNUC__ >= 4
#define PGM_VNF_CONVERTER_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define PGM_VNF_CONVERTER_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define PGM_VNF_CONVERTER_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define PGM_VNF_CONVERTER_HELPER_DLL_IMPORT
#define PGM_VNF_CONVERTER_HELPER_DLL_EXPORT
#define PGM_VNF_CONVERTER_HELPER_DLL_LOCAL
#endif
#endif
// Now we use the generic helper definitions above to define PGM_API and PGM_LOCAL.
#ifdef PGM_DLL_EXPORTS // defined if we are building the POWER_GRID_MODEL DLL (instead of using it)
#define PGM_VNF_converter_API PGM_VNF_CONVERTER_HELPER_DLL_EXPORT
#else
#define PGM_VNF_converter_API PGM_VNF_CONVERTER_HELPER_DLL_IMPORT

// API_MACRO_BLOCK

#endif
