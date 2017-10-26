message(STATUS "downloading...
     src='http://downloads.webmproject.org/releases/webp/libwebp-0.6.0.tar.gz'
     dst='/misc/student/muazzama/demon/lmbspecialops/build/webp/src/libwebp-0.6.0.tar.gz'
     timeout='none'")




file(DOWNLOAD
  "http://downloads.webmproject.org/releases/webp/libwebp-0.6.0.tar.gz"
  "/misc/student/muazzama/demon/lmbspecialops/build/webp/src/libwebp-0.6.0.tar.gz"
  SHOW_PROGRESS
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'http://downloads.webmproject.org/releases/webp/libwebp-0.6.0.tar.gz' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
