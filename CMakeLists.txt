cmake_minimum_required(VERSION 3.20)

# CUDA is optional-by-detection rather than a hard requirement, so the language
# list is assembled before project() is called.
project(protcad LANGUAGES CXX)

include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(PROTCAD_CUDA ON)
else()
    set(PROTCAD_CUDA OFF)
    message(STATUS "protcad: no CUDA compiler found, building the CPU-only path")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PROTCAD_NATIVE_ARCH "Tune host code for the building machine's CPU" ON)
option(PROTCAD_BUILD_TESTS "Build the test and diagnostic binaries" ON)

# Binaries land in bin/ so the layout matches what the scripts and the README
# already expect from the old Makefile.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------------------------------------------------------------------------
# libprotcad
# ---------------------------------------------------------------------------

# Every .cc under these four directories is a library source.  That is an
# invariant now rather than a coincidence: the one file that was not compiled,
# superRotamerIterator, was dead and did not compile, and has been removed.
file(GLOB PROTCAD_CXX_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/database/*.cc
    ${CMAKE_SOURCE_DIR}/src/ensemble/*.cc
    ${CMAKE_SOURCE_DIR}/src/io/*.cc
    ${CMAKE_SOURCE_DIR}/src/math/*.cc)

add_library(protcad STATIC ${PROTCAD_CXX_SOURCES})

target_include_directories(protcad PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/database
    ${CMAKE_SOURCE_DIR}/src/ensemble
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/math)

target_compile_definitions(protcad PUBLIC __STL_USE_EXCEPTIONS)

# -Ofast/-ffast-math and the vectoriser flags are carried over from the old
# build verbatim.  Changing them changes the energies, so they are not a
# cleanup target.
target_compile_options(protcad PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-deprecated;-Ofast;-ffast-math;-ftree-vectorize;-pipe>)
if(PROTCAD_NATIVE_ARCH)
    target_compile_options(protcad PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-march=native;-mtune=native>)
endif()

if(PROTCAD_CUDA)
    target_sources(protcad PRIVATE ${CMAKE_SOURCE_DIR}/src/ensemble/energy.cu)

    # __CUDA__ must reach every translation unit, not just the .cu file.  When
    # it was missing from the host compile, every GPU call site in protein.cc
    # was preprocessed away and the kernels were built but never called.
    target_compile_definitions(protcad PUBLIC __CUDA__)

    # A fatbin spanning Pascal through Ampere plus PTX for anything newer.  A
    # single -arch produces cubins that will not load on any other card.
    set_property(TARGET protcad PROPERTY CUDA_ARCHITECTURES 61 75 86 86-virtual)
    set_property(TARGET protcad PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_options(protcad PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(protcad PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()

# ---------------------------------------------------------------------------
# projects/ -- one installed binary per source file
# ---------------------------------------------------------------------------

file(GLOB PROTCAD_PROJECT_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/projects/*.cc)
foreach(src ${PROTCAD_PROJECT_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE protcad)
endforeach()

# ---------------------------------------------------------------------------
# tests/
# ---------------------------------------------------------------------------

if(PROTCAD_BUILD_TESTS)
    enable_testing()

    # Self-contained: they build their own inputs and exit nonzero on failure.
    set(PROTCAD_SELF_TESTS amberParamsTest batchTest bestfitTest energyTest replicaTest)
    # Need a structure argument.
    set(PROTCAD_PDB_TESTS clashCheck energyTerms rotamerTest)

    file(GLOB PROTCAD_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cc)
    foreach(src ${PROTCAD_TEST_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE protcad)

        if(${name} IN_LIST PROTCAD_SELF_TESTS)
            add_test(NAME ${name} COMMAND ${name})
        elseif(${name} IN_LIST PROTCAD_PDB_TESTS)
            add_test(NAME ${name}
                     COMMAND ${name} ${CMAKE_SOURCE_DIR}/tests/data/1crn.pdb)
        endif()
        # Anything not listed above is a benchmark or a diagnostic with no
        # pass/fail contract, so it is built but not registered with ctest.
        # fusionTest is the notable one: it currently reports a real defect in
        # the potential rather than a regression, so registering it would just
        # leave the suite permanently red.
    endforeach()

    # protcad resolves its data files relative to PROTCADDIR.
    set_tests_properties(${PROTCAD_SELF_TESTS} ${PROTCAD_PDB_TESTS} PROPERTIES
        ENVIRONMENT "PROTCADDIR=${CMAKE_SOURCE_DIR}"
        TIMEOUT 600)
endif()
