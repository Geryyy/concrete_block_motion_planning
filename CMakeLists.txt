cmake_minimum_required(VERSION 3.8)
project(concrete_block_motion_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/grip_traj_server_simple.py
  scripts/grip_trajectory.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  motion_planning
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_grip_trajectory test/test_grip_trajectory.py)
endif()

ament_package()
