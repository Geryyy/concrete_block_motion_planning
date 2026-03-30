cmake_minimum_required(VERSION 3.8)
project(concrete_block_motion_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(concrete_block_perception REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanGeometricPath.srv"
  "srv/ComputeTrajectory.srv"
  "srv/ExecuteTrajectory.srv"
  "srv/ExecuteNamedConfiguration.srv"
  "srv/GetNextAssemblyTask.srv"
  "srv/PlanAndComputeTrajectory.srv"
  DEPENDENCIES geometry_msgs nav_msgs std_msgs builtin_interfaces trajectory_msgs
)

install(PROGRAMS
  scripts/motion_planning_node.py
  scripts/acados_benchmark.py
  scripts/bare_metal_single_block_client.py
  scripts/headless_trajectory_smoke.py
  scripts/joint_goal_feasibility_probe.py
  scripts/rviz_move_empty_interface.py
  scripts/joint_state_timber_compat.py
  scripts/dummy_joint_state_publisher.py
  scripts/timber_follow_joint_trajectory_proxy.py
  scripts/dummy_gripper_state_publisher.py
  scripts/gripper_state_bridge.py
  motion_planning_tools/standalone/run_planner_experiment.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  scripts/cbmp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  motion_planning
  motion_planning_tools
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  VENDOR_README.md
  pyproject.toml
  requirements-core.txt
  requirements-benchmark.txt
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
