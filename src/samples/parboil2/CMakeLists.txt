include_directories(
    common/include
)

add_subdirectory(bfs)
add_subdirectory(cutcp)
add_subdirectory(histo)
add_subdirectory(lbm)
add_subdirectory(mri-gridding)
add_subdirectory(mri-q)
add_subdirectory(sad)
add_subdirectory(sgemm)
add_subdirectory(spmv)
add_subdirectory(stencil)
add_subdirectory(tpacf)

# GLoop versions.
add_subdirectory(lbm-gloop)
add_subdirectory(tpacf-gloop)
add_subdirectory(stencil-gloop)
add_subdirectory(mri-gridding-gloop)
add_subdirectory(mri-q-gloop)

# GLoop naive version.
add_subdirectory(tpacf-naive-gloop)

# Split kernels versions.
add_subdirectory(tpacf-kernel)
