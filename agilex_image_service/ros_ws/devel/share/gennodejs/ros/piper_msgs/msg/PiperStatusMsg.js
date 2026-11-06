// Auto-generated. Do not edit!

// (in-package piper_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class PiperStatusMsg {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.ctrl_mode = null;
      this.arm_status = null;
      this.mode_feedback = null;
      this.teach_status = null;
      this.motion_status = null;
      this.trajectory_num = null;
      this.err_code = null;
      this.joint_1_angle_limit = null;
      this.joint_2_angle_limit = null;
      this.joint_3_angle_limit = null;
      this.joint_4_angle_limit = null;
      this.joint_5_angle_limit = null;
      this.joint_6_angle_limit = null;
      this.communication_status_joint_1 = null;
      this.communication_status_joint_2 = null;
      this.communication_status_joint_3 = null;
      this.communication_status_joint_4 = null;
      this.communication_status_joint_5 = null;
      this.communication_status_joint_6 = null;
    }
    else {
      if (initObj.hasOwnProperty('ctrl_mode')) {
        this.ctrl_mode = initObj.ctrl_mode
      }
      else {
        this.ctrl_mode = 0;
      }
      if (initObj.hasOwnProperty('arm_status')) {
        this.arm_status = initObj.arm_status
      }
      else {
        this.arm_status = 0;
      }
      if (initObj.hasOwnProperty('mode_feedback')) {
        this.mode_feedback = initObj.mode_feedback
      }
      else {
        this.mode_feedback = 0;
      }
      if (initObj.hasOwnProperty('teach_status')) {
        this.teach_status = initObj.teach_status
      }
      else {
        this.teach_status = 0;
      }
      if (initObj.hasOwnProperty('motion_status')) {
        this.motion_status = initObj.motion_status
      }
      else {
        this.motion_status = 0;
      }
      if (initObj.hasOwnProperty('trajectory_num')) {
        this.trajectory_num = initObj.trajectory_num
      }
      else {
        this.trajectory_num = 0;
      }
      if (initObj.hasOwnProperty('err_code')) {
        this.err_code = initObj.err_code
      }
      else {
        this.err_code = 0;
      }
      if (initObj.hasOwnProperty('joint_1_angle_limit')) {
        this.joint_1_angle_limit = initObj.joint_1_angle_limit
      }
      else {
        this.joint_1_angle_limit = false;
      }
      if (initObj.hasOwnProperty('joint_2_angle_limit')) {
        this.joint_2_angle_limit = initObj.joint_2_angle_limit
      }
      else {
        this.joint_2_angle_limit = false;
      }
      if (initObj.hasOwnProperty('joint_3_angle_limit')) {
        this.joint_3_angle_limit = initObj.joint_3_angle_limit
      }
      else {
        this.joint_3_angle_limit = false;
      }
      if (initObj.hasOwnProperty('joint_4_angle_limit')) {
        this.joint_4_angle_limit = initObj.joint_4_angle_limit
      }
      else {
        this.joint_4_angle_limit = false;
      }
      if (initObj.hasOwnProperty('joint_5_angle_limit')) {
        this.joint_5_angle_limit = initObj.joint_5_angle_limit
      }
      else {
        this.joint_5_angle_limit = false;
      }
      if (initObj.hasOwnProperty('joint_6_angle_limit')) {
        this.joint_6_angle_limit = initObj.joint_6_angle_limit
      }
      else {
        this.joint_6_angle_limit = false;
      }
      if (initObj.hasOwnProperty('communication_status_joint_1')) {
        this.communication_status_joint_1 = initObj.communication_status_joint_1
      }
      else {
        this.communication_status_joint_1 = false;
      }
      if (initObj.hasOwnProperty('communication_status_joint_2')) {
        this.communication_status_joint_2 = initObj.communication_status_joint_2
      }
      else {
        this.communication_status_joint_2 = false;
      }
      if (initObj.hasOwnProperty('communication_status_joint_3')) {
        this.communication_status_joint_3 = initObj.communication_status_joint_3
      }
      else {
        this.communication_status_joint_3 = false;
      }
      if (initObj.hasOwnProperty('communication_status_joint_4')) {
        this.communication_status_joint_4 = initObj.communication_status_joint_4
      }
      else {
        this.communication_status_joint_4 = false;
      }
      if (initObj.hasOwnProperty('communication_status_joint_5')) {
        this.communication_status_joint_5 = initObj.communication_status_joint_5
      }
      else {
        this.communication_status_joint_5 = false;
      }
      if (initObj.hasOwnProperty('communication_status_joint_6')) {
        this.communication_status_joint_6 = initObj.communication_status_joint_6
      }
      else {
        this.communication_status_joint_6 = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PiperStatusMsg
    // Serialize message field [ctrl_mode]
    bufferOffset = _serializer.uint8(obj.ctrl_mode, buffer, bufferOffset);
    // Serialize message field [arm_status]
    bufferOffset = _serializer.uint8(obj.arm_status, buffer, bufferOffset);
    // Serialize message field [mode_feedback]
    bufferOffset = _serializer.uint8(obj.mode_feedback, buffer, bufferOffset);
    // Serialize message field [teach_status]
    bufferOffset = _serializer.uint8(obj.teach_status, buffer, bufferOffset);
    // Serialize message field [motion_status]
    bufferOffset = _serializer.uint8(obj.motion_status, buffer, bufferOffset);
    // Serialize message field [trajectory_num]
    bufferOffset = _serializer.uint8(obj.trajectory_num, buffer, bufferOffset);
    // Serialize message field [err_code]
    bufferOffset = _serializer.int64(obj.err_code, buffer, bufferOffset);
    // Serialize message field [joint_1_angle_limit]
    bufferOffset = _serializer.bool(obj.joint_1_angle_limit, buffer, bufferOffset);
    // Serialize message field [joint_2_angle_limit]
    bufferOffset = _serializer.bool(obj.joint_2_angle_limit, buffer, bufferOffset);
    // Serialize message field [joint_3_angle_limit]
    bufferOffset = _serializer.bool(obj.joint_3_angle_limit, buffer, bufferOffset);
    // Serialize message field [joint_4_angle_limit]
    bufferOffset = _serializer.bool(obj.joint_4_angle_limit, buffer, bufferOffset);
    // Serialize message field [joint_5_angle_limit]
    bufferOffset = _serializer.bool(obj.joint_5_angle_limit, buffer, bufferOffset);
    // Serialize message field [joint_6_angle_limit]
    bufferOffset = _serializer.bool(obj.joint_6_angle_limit, buffer, bufferOffset);
    // Serialize message field [communication_status_joint_1]
    bufferOffset = _serializer.bool(obj.communication_status_joint_1, buffer, bufferOffset);
    // Serialize message field [communication_status_joint_2]
    bufferOffset = _serializer.bool(obj.communication_status_joint_2, buffer, bufferOffset);
    // Serialize message field [communication_status_joint_3]
    bufferOffset = _serializer.bool(obj.communication_status_joint_3, buffer, bufferOffset);
    // Serialize message field [communication_status_joint_4]
    bufferOffset = _serializer.bool(obj.communication_status_joint_4, buffer, bufferOffset);
    // Serialize message field [communication_status_joint_5]
    bufferOffset = _serializer.bool(obj.communication_status_joint_5, buffer, bufferOffset);
    // Serialize message field [communication_status_joint_6]
    bufferOffset = _serializer.bool(obj.communication_status_joint_6, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PiperStatusMsg
    let len;
    let data = new PiperStatusMsg(null);
    // Deserialize message field [ctrl_mode]
    data.ctrl_mode = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [arm_status]
    data.arm_status = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [mode_feedback]
    data.mode_feedback = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [teach_status]
    data.teach_status = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [motion_status]
    data.motion_status = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [trajectory_num]
    data.trajectory_num = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [err_code]
    data.err_code = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [joint_1_angle_limit]
    data.joint_1_angle_limit = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [joint_2_angle_limit]
    data.joint_2_angle_limit = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [joint_3_angle_limit]
    data.joint_3_angle_limit = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [joint_4_angle_limit]
    data.joint_4_angle_limit = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [joint_5_angle_limit]
    data.joint_5_angle_limit = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [joint_6_angle_limit]
    data.joint_6_angle_limit = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [communication_status_joint_1]
    data.communication_status_joint_1 = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [communication_status_joint_2]
    data.communication_status_joint_2 = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [communication_status_joint_3]
    data.communication_status_joint_3 = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [communication_status_joint_4]
    data.communication_status_joint_4 = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [communication_status_joint_5]
    data.communication_status_joint_5 = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [communication_status_joint_6]
    data.communication_status_joint_6 = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 26;
  }

  static datatype() {
    // Returns string type for a message object
    return 'piper_msgs/PiperStatusMsg';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '18e0aaa943372aaa58f1495907dd9a17';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 ctrl_mode
    uint8 arm_status
    uint8 mode_feedback
    uint8 teach_status
    uint8 motion_status
    uint8 trajectory_num
    int64 err_code
    bool joint_1_angle_limit
    bool joint_2_angle_limit
    bool joint_3_angle_limit
    bool joint_4_angle_limit
    bool joint_5_angle_limit
    bool joint_6_angle_limit
    bool communication_status_joint_1
    bool communication_status_joint_2
    bool communication_status_joint_3
    bool communication_status_joint_4
    bool communication_status_joint_5
    bool communication_status_joint_6
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PiperStatusMsg(null);
    if (msg.ctrl_mode !== undefined) {
      resolved.ctrl_mode = msg.ctrl_mode;
    }
    else {
      resolved.ctrl_mode = 0
    }

    if (msg.arm_status !== undefined) {
      resolved.arm_status = msg.arm_status;
    }
    else {
      resolved.arm_status = 0
    }

    if (msg.mode_feedback !== undefined) {
      resolved.mode_feedback = msg.mode_feedback;
    }
    else {
      resolved.mode_feedback = 0
    }

    if (msg.teach_status !== undefined) {
      resolved.teach_status = msg.teach_status;
    }
    else {
      resolved.teach_status = 0
    }

    if (msg.motion_status !== undefined) {
      resolved.motion_status = msg.motion_status;
    }
    else {
      resolved.motion_status = 0
    }

    if (msg.trajectory_num !== undefined) {
      resolved.trajectory_num = msg.trajectory_num;
    }
    else {
      resolved.trajectory_num = 0
    }

    if (msg.err_code !== undefined) {
      resolved.err_code = msg.err_code;
    }
    else {
      resolved.err_code = 0
    }

    if (msg.joint_1_angle_limit !== undefined) {
      resolved.joint_1_angle_limit = msg.joint_1_angle_limit;
    }
    else {
      resolved.joint_1_angle_limit = false
    }

    if (msg.joint_2_angle_limit !== undefined) {
      resolved.joint_2_angle_limit = msg.joint_2_angle_limit;
    }
    else {
      resolved.joint_2_angle_limit = false
    }

    if (msg.joint_3_angle_limit !== undefined) {
      resolved.joint_3_angle_limit = msg.joint_3_angle_limit;
    }
    else {
      resolved.joint_3_angle_limit = false
    }

    if (msg.joint_4_angle_limit !== undefined) {
      resolved.joint_4_angle_limit = msg.joint_4_angle_limit;
    }
    else {
      resolved.joint_4_angle_limit = false
    }

    if (msg.joint_5_angle_limit !== undefined) {
      resolved.joint_5_angle_limit = msg.joint_5_angle_limit;
    }
    else {
      resolved.joint_5_angle_limit = false
    }

    if (msg.joint_6_angle_limit !== undefined) {
      resolved.joint_6_angle_limit = msg.joint_6_angle_limit;
    }
    else {
      resolved.joint_6_angle_limit = false
    }

    if (msg.communication_status_joint_1 !== undefined) {
      resolved.communication_status_joint_1 = msg.communication_status_joint_1;
    }
    else {
      resolved.communication_status_joint_1 = false
    }

    if (msg.communication_status_joint_2 !== undefined) {
      resolved.communication_status_joint_2 = msg.communication_status_joint_2;
    }
    else {
      resolved.communication_status_joint_2 = false
    }

    if (msg.communication_status_joint_3 !== undefined) {
      resolved.communication_status_joint_3 = msg.communication_status_joint_3;
    }
    else {
      resolved.communication_status_joint_3 = false
    }

    if (msg.communication_status_joint_4 !== undefined) {
      resolved.communication_status_joint_4 = msg.communication_status_joint_4;
    }
    else {
      resolved.communication_status_joint_4 = false
    }

    if (msg.communication_status_joint_5 !== undefined) {
      resolved.communication_status_joint_5 = msg.communication_status_joint_5;
    }
    else {
      resolved.communication_status_joint_5 = false
    }

    if (msg.communication_status_joint_6 !== undefined) {
      resolved.communication_status_joint_6 = msg.communication_status_joint_6;
    }
    else {
      resolved.communication_status_joint_6 = false
    }

    return resolved;
    }
};

module.exports = PiperStatusMsg;
