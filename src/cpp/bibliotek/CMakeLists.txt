add_library(bibliotek STATIC
            ferrlog.cpp
            GetOpt.cpp
            outils.cpp
            primitive.cpp
            T_geometrie.cpp
            getallfilename.cpp
            decodeclef.cpp
            chrono.cpp)

if (WIN32)
    target_compile_definitions(meschach PUBLIC MINGW)
    target_compile_definitions(meschach PUBLIC WIN32)
endif()


target_include_directories(bibliotek PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/include)