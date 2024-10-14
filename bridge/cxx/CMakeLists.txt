cmake_minimum_required(VERSION 3.21)
project(cxxbridge-cxx)

set(CMAKE_CXX_STANDARD 17)

# set(CARGO_BUILD_TARGET ${CARGO_TARGET_DIR})

message(STATUS "------" ${CARGO_BUILD_TARGET})

set(CARGO_MANIFEST ${CMAKE_SOURCE_DIR}/../Cargo.toml)
set(CARGO_TARGET_DIR ${CMAKE_SOURCE_DIR}/../../target)
set(RUST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../src)
set(RUST_SOURCE_FILES 
    ${RUST_SOURCE_DIR}/nndevice/ffi.rs
    )
set(CXX_SOURCE_FILES
    src/nndevice.cc
    )
set(RUST_BRIDGE_CXX 
    ${CARGO_TARGET_DIR}/${CARGO_BUILD_TARGET}/cxxbridge/bridge/src/nndevice/ffi.rs.cc
    )
# set(RUST_LIB ${CARGO_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}bridge${CMAKE_STATIC_LIBRARY_SUFFIX})

if(!FROM_CARGO)
    # Add a custom command that builds the rust crate and generates C++ bridge code
    add_custom_command(
        OUTPUT ${RUST_BRIDGE_CXX}
        COMMAND cargo build --manifest-path ${CARGO_MANIFEST}
        DEPENDS ${RUST_SOURCE_FILES}
        USES_TERMINAL
        COMMENT "Running cargo..."
    )
endif()

add_library (cxxbridge-cxx STATIC ${CXX_SOURCE_FILES} ${RUST_BRIDGE_CXX})

target_include_directories(
    cxxbridge-cxx
    PRIVATE
    include/
    ${CARGO_TARGET_DIR}/${CARGO_BUILD_TARGET}/cxxbridge
    ${CARGO_TARGET_DIR}/${CARGO_BUILD_TARGET}/cxxbridge/bridge/src/
)

target_link_libraries(cxxbridge-cxx)

set_target_properties(
    cxxbridge-cxx
    PROPERTIES ADDITIONAL_CLEAN_FILES ${CARGO_TARGET_DIR}
)

install (TARGETS cxxbridge-cxx DESTINATION .)
