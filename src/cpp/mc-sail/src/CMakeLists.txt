add_executable(mcsail
                msail.cpp
                multicou.cpp
                profil.cpp)

target_include_directories(mcsail PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/include)
target_include_directories(mcsail PUBLIC ${PROJECT_SOURCE_DIR}/src/cpp/meschach/mesch12a/include)

target_link_libraries(mcsail PUBLIC bibliotek meschach)
