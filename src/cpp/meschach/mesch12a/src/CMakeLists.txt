# Collect all C source files in this directory
file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

# Build meschach as a static library
add_library(meschach STATIC ${SRC_FILES})

# Windows-specific defines
if (WIN32)
    target_compile_definitions(meschach PUBLIC MINGW WIN32)
endif()

# Force old C standard (gnu89) for meschach sources only
set_source_files_properties(${SRC_FILES} PROPERTIES COMPILE_FLAGS "-std=gnu89")

# Make sure the compiler treats it as C
set_target_properties(meschach PROPERTIES LINKER_LANGUAGE C)

# Public include path (existing headers in meschach/include)
target_include_directories(meschach
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src/cpp/meschach/mesch12a/include
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}         # so machine.h is visible
        ${CMAKE_BINARY_DIR}                 # root-level binary dir where machine.h lives
)
