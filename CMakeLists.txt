cmake_minimum_required(VERSION 3.8)
project(concrete_block_motion_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetNextAssemblyTask.srv"
  DEPENDENCIES geometry_msgs builtin_interfaces
)

install(PROGRAMS
  scripts/wall_plan_server.py
  scripts/grip_traj_server_simple.py
  scripts/grip_trajectory.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  motion_planning
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
