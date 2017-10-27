message(STATUS "downloading...
     src='https://github.com/lz4/lz4/archive/r131.tar.gz'
     dst='/misc/student/muazzama/demon/lmbspecialops/build/lz4/src/r131.tar.gz'
     timeout='none'")




file(DOWNLOAD
  "https://github.com/lz4/lz4/archive/r131.tar.gz"
  "/misc/student/muazzama/demon/lmbspecialops/build/lz4/src/r131.tar.gz"
  SHOW_PROGRESS
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'https://github.com/lz4/lz4/archive/r131.tar.gz' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
