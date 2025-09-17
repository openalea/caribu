add_executable(s2v
                s2v.cpp
                main.cpp)

target_include_directories(s2v PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/include)
target_include_directories(s2v PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/meschach/mesch12a/include)

target_link_libraries(s2v PUBLIC bibliotek meschach)