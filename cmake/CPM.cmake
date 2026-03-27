# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 Lars Melchior and contributors

set(CPM_DOWNLOAD_VERSION 0.42.1)
set(CPM_HASH_SUM "f3a6dcc6a04ce9e7f51a127307fa4f699fb2bade357a8eb4c5b45df76e1dc6a5")

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

set(CPM_LOCAL_FALLBACKS
    "${CMAKE_SOURCE_DIR}/build/default/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake"
)

set(CPM_HAVE_VALID_LOCAL_COPY FALSE)
if(EXISTS "${CPM_DOWNLOAD_LOCATION}")
  file(SHA256 "${CPM_DOWNLOAD_LOCATION}" CPM_EXISTING_HASH)
  if(CPM_EXISTING_HASH STREQUAL CPM_HASH_SUM)
    set(CPM_HAVE_VALID_LOCAL_COPY TRUE)
  endif()
endif()

if(NOT CPM_HAVE_VALID_LOCAL_COPY)
  foreach(CPM_FALLBACK ${CPM_LOCAL_FALLBACKS})
    if(EXISTS "${CPM_FALLBACK}")
      file(SHA256 "${CPM_FALLBACK}" CPM_FALLBACK_HASH)
      if(CPM_FALLBACK_HASH STREQUAL CPM_HASH_SUM)
        file(COPY_FILE "${CPM_FALLBACK}" "${CPM_DOWNLOAD_LOCATION}" ONLY_IF_DIFFERENT)
        set(CPM_HAVE_VALID_LOCAL_COPY TRUE)
        break()
      endif()
    endif()
  endforeach()
endif()

if(NOT CPM_HAVE_VALID_LOCAL_COPY)
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=${CPM_HASH_SUM}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})
