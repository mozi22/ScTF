# CMake generated Testfile for 
# Source directory: /misc/student/muazzama/demon/lmbspecialops/test
# Build directory: /misc/student/muazzama/demon/lmbspecialops/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_FloOp "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_FloOp.py")
set_tests_properties(test_FloOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_Lz4 "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_Lz4.py")
set_tests_properties(test_Lz4 PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_ReplaceNonfinite "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_ReplaceNonfinite.py")
set_tests_properties(test_ReplaceNonfinite PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_FlowToDepth2 "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_FlowToDepth2.py")
set_tests_properties(test_FlowToDepth2 PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_Lz4Raw "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_Lz4Raw.py")
set_tests_properties(test_Lz4Raw PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_ScaleInvariantGradient "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_ScaleInvariantGradient.py")
set_tests_properties(test_ScaleInvariantGradient PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_Median3x3Downsample "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_Median3x3Downsample.py")
set_tests_properties(test_Median3x3Downsample PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_WebpOp "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_WebpOp.py")
set_tests_properties(test_WebpOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_PpmOp "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_PpmOp.py")
set_tests_properties(test_PpmOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_LeakyRelu "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_LeakyRelu.py")
set_tests_properties(test_LeakyRelu PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
add_test(test_PfmOp "/misc/student/muazzama/virtualenv/thesis/bin/python" "test_PfmOp.py")
set_tests_properties(test_PfmOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/misc/student/muazzama/demon/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/misc/student/muazzama/demon/lmbspecialops/test")
