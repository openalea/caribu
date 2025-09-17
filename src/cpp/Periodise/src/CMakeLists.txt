
add_executable(periodise
                actop.cpp
                diffuseur.cpp
                canopyL.cpp
                GetOpt.cpp
                outils.cpp
                T_geometrie.cpp
                primitive.cpp
                periodise.cpp
                main.cpp)

target_include_directories(periodise PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/include)
target_include_directories(periodise PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/meschach/mesch12a/include)

target_link_libraries(periodise PUBLIC bibliotek meschach)