cmake_minimum_required(VERSION 3.20)
project(caribu LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories("src/cpp/")

# -------------------------------------------------------
# === Meschach compatibility checks ===
# These replicate what the old configure.in used to do
# -------------------------------------------------------
include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckTypeSize)
include(CheckCSourceCompiles)
include(TestBigEndian)

# Headers
check_include_file("memory.h"   HAVE_MEMORY_H)
check_include_file("malloc.h"   HAVE_MALLOC_H)
check_include_file("complex.h"  HAVE_COMPLEX_H)
check_include_file("varargs.h"  HAVE_VARARGS_H)
check_include_file("unistd.h"   HAVE_UNISTD_H)

# Functions
check_function_exists(bcopy HAVE_BCOPY)
check_function_exists(bzero HAVE_BZERO)
check_function_exists(memcpy HAVE_MEMCPY)

# Types & sizes
check_type_size("int" SIZEOF_INT)
check_type_size("long" SIZEOF_LONG)
check_type_size("double" SIZEOF_DOUBLE)
check_type_size("size_t" SIZEOF_SIZE_T)

# u_int typedef check
check_c_source_compiles("
  #include <stdio.h>
  #include <stdlib.h>
  int main() { u_int i = 1; return 0; }
" U_INT_DEF)

# const keyword
check_c_source_compiles("
  int f(const int x) { return x; }
  int main(){return f(1);}
" HAVE_CONST)

# Prototypes
check_c_source_compiles("
  extern int test (int i, double x);
  int main() { return 0; }
" HAVE_PROTOTYPES)

check_c_source_compiles("
  struct s1 {int (*f) (int a);};
  struct s2 {int (*f) (double a);};
  int main() { return 0; }
" HAVE_PROTOTYPES_IN_STRUCT)

# -------------------------------------------------------
# Define ANSI_C for modern compilers
# -------------------------------------------------------
check_c_source_compiles("
  int main(void) { return 0; }
" HAVE_STANDARD_C)

if(HAVE_STANDARD_C)
  add_definitions(-DANSI_C)
endif()

# Endianness
test_big_endian(WORDS_BIGENDIAN)

# Options (instead of --with-* in Autoconf)
option(WITH_COMPLEX "Use complex functions" OFF)
option(WITH_SPARSE  "Use sparse matrix functions" OFF)
option(WITH_UNROLL  "Unroll vector loops" OFF)
option(WITH_MUNROLL "Unroll matrix loops" OFF)
option(WITH_FLOAT   "Use single precision" OFF)
option(WITH_DOUBLE  "Use double precision" ON)

if(WITH_COMPLEX)
  add_compile_definitions(COMPLEX)
endif()
if(WITH_SPARSE)
  add_compile_definitions(SPARSE)
endif()
if(WITH_UNROLL)
  add_compile_definitions(VUNROLL)
endif()
if(WITH_MUNROLL)
  add_compile_definitions(MUNROLL)
endif()
if(WITH_FLOAT)
  add_compile_definitions(REAL_FLT)
endif()
if(WITH_DOUBLE)
  add_compile_definitions(REAL_DBL)
endif()

# Special constants
# Write a small C program to compute constants
file(WRITE ${CMAKE_BINARY_DIR}/check_constants.c "
#include <stdio.h>
#include <float.h>
#include <limits.h>
int main(void) {
    printf(\"D_MACHEPS=%.20g\\n\", DBL_EPSILON);
    printf(\"F_MACHEPS=%.9gF\\n\", FLT_EPSILON);
    printf(\"M_MAX_INT=%d\\n\", INT_MAX);
    return 0;
}
")

# Compile and run it
try_run(RUN_RESULT COMPILE_RESULT
    ${CMAKE_BINARY_DIR}        # run directory
    ${CMAKE_BINARY_DIR}/check_constants.c
    COMPILE_OUTPUT_VARIABLE COMPILE_OUT
    RUN_OUTPUT_VARIABLE OUTPUT_STR
)

if(COMPILE_RESULT AND RUN_RESULT EQUAL 0)
    string(REGEX MATCH "D_MACHEPS=([0-9.e+-]+)" _match "${OUTPUT_STR}")
    set(D_MACHEPS "${CMAKE_MATCH_1}")
    string(REGEX MATCH "F_MACHEPS=([0-9.e+-]+F)" _match "${OUTPUT_STR}")
    set(F_MACHEPS "${CMAKE_MATCH_1}")
    string(REGEX MATCH "M_MAX_INT=([0-9]+)" _match "${OUTPUT_STR}")
    set(M_MAX_INT "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "Could not compute machine constants: ${COMPILE_OUT}")
endif()


# Generate machine.h in build dir
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/machine.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/machine.h
)

# -------------------------------------------------------
# === Project subdirectories ===
# -------------------------------------------------------
add_subdirectory("src/cpp/meschach/mesch12a/src")
add_subdirectory("src/cpp/bibliotek")
add_subdirectory("src/cpp/mc-sail/src")
add_subdirectory("src/cpp/Periodise/src")
add_subdirectory("src/cpp/s2v/src")
add_subdirectory("src/cpp/Canestra/src")

# -------------------------------------------------------
# === Installation rules ===
# -------------------------------------------------------
install(TARGETS mcsail periodise s2v canestrad
        DESTINATION "$ENV{PREFIX}/bin")

