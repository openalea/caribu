add_executable(canestrad
            ff.cpp
            bsp.cpp
            bzh.cpp
            diffuseur.cpp
            infini.cpp
            solver.cpp
            radioxity.cpp
            voxel.cpp
            canopy_E.cpp
            canopy_io.cpp)
target_compile_definitions(canestrad PUBLIC _HD)

if (WIN32)
    target_compile_definitions(meschach PUBLIC MINGW)
    target_compile_definitions(meschach PUBLIC WIN32)
endif()

target_include_directories(canestrad PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/include)
target_include_directories(canestrad PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/meschach/mesch12a/include)

target_link_libraries(canestrad PUBLIC meschach bibliotek)