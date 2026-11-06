# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "piper_msgs: 2 messages, 0 services")

set(MSG_I_FLAGS "-Ipiper_msgs:/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(piper_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg" NAME_WE)
add_custom_target(_piper_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "piper_msgs" "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg" ""
)

get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg" NAME_WE)
add_custom_target(_piper_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "piper_msgs" "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/piper_msgs
)
_generate_msg_cpp(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/piper_msgs
)

### Generating Services

### Generating Module File
_generate_module_cpp(piper_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/piper_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(piper_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(piper_msgs_generate_messages piper_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_cpp _piper_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_cpp _piper_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(piper_msgs_gencpp)
add_dependencies(piper_msgs_gencpp piper_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS piper_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/piper_msgs
)
_generate_msg_eus(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/piper_msgs
)

### Generating Services

### Generating Module File
_generate_module_eus(piper_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/piper_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(piper_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(piper_msgs_generate_messages piper_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_eus _piper_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_eus _piper_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(piper_msgs_geneus)
add_dependencies(piper_msgs_geneus piper_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS piper_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/piper_msgs
)
_generate_msg_lisp(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/piper_msgs
)

### Generating Services

### Generating Module File
_generate_module_lisp(piper_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/piper_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(piper_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(piper_msgs_generate_messages piper_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_lisp _piper_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_lisp _piper_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(piper_msgs_genlisp)
add_dependencies(piper_msgs_genlisp piper_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS piper_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/piper_msgs
)
_generate_msg_nodejs(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/piper_msgs
)

### Generating Services

### Generating Module File
_generate_module_nodejs(piper_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/piper_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(piper_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(piper_msgs_generate_messages piper_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_nodejs _piper_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_nodejs _piper_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(piper_msgs_gennodejs)
add_dependencies(piper_msgs_gennodejs piper_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS piper_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/piper_msgs
)
_generate_msg_py(piper_msgs
  "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/piper_msgs
)

### Generating Services

### Generating Module File
_generate_module_py(piper_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/piper_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(piper_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(piper_msgs_generate_messages piper_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PiperStatusMsg.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_py _piper_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jchen572/code_file/agilex_image_service/ros_ws/src/piper_msgs/msg/PosCmd.msg" NAME_WE)
add_dependencies(piper_msgs_generate_messages_py _piper_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(piper_msgs_genpy)
add_dependencies(piper_msgs_genpy piper_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS piper_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/piper_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/piper_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_cpp)
  add_dependencies(piper_msgs_generate_messages_cpp actionlib_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(piper_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/piper_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/piper_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_eus)
  add_dependencies(piper_msgs_generate_messages_eus actionlib_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(piper_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/piper_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/piper_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_lisp)
  add_dependencies(piper_msgs_generate_messages_lisp actionlib_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(piper_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/piper_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/piper_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_nodejs)
  add_dependencies(piper_msgs_generate_messages_nodejs actionlib_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(piper_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/piper_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/piper_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/piper_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_py)
  add_dependencies(piper_msgs_generate_messages_py actionlib_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(piper_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
