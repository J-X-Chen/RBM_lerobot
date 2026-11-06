; Auto-generated. Do not edit!


(cl:in-package piper_msgs-msg)


;//! \htmlinclude PiperStatusMsg.msg.html

(cl:defclass <PiperStatusMsg> (roslisp-msg-protocol:ros-message)
  ((ctrl_mode
    :reader ctrl_mode
    :initarg :ctrl_mode
    :type cl:fixnum
    :initform 0)
   (arm_status
    :reader arm_status
    :initarg :arm_status
    :type cl:fixnum
    :initform 0)
   (mode_feedback
    :reader mode_feedback
    :initarg :mode_feedback
    :type cl:fixnum
    :initform 0)
   (teach_status
    :reader teach_status
    :initarg :teach_status
    :type cl:fixnum
    :initform 0)
   (motion_status
    :reader motion_status
    :initarg :motion_status
    :type cl:fixnum
    :initform 0)
   (trajectory_num
    :reader trajectory_num
    :initarg :trajectory_num
    :type cl:fixnum
    :initform 0)
   (err_code
    :reader err_code
    :initarg :err_code
    :type cl:integer
    :initform 0)
   (joint_1_angle_limit
    :reader joint_1_angle_limit
    :initarg :joint_1_angle_limit
    :type cl:boolean
    :initform cl:nil)
   (joint_2_angle_limit
    :reader joint_2_angle_limit
    :initarg :joint_2_angle_limit
    :type cl:boolean
    :initform cl:nil)
   (joint_3_angle_limit
    :reader joint_3_angle_limit
    :initarg :joint_3_angle_limit
    :type cl:boolean
    :initform cl:nil)
   (joint_4_angle_limit
    :reader joint_4_angle_limit
    :initarg :joint_4_angle_limit
    :type cl:boolean
    :initform cl:nil)
   (joint_5_angle_limit
    :reader joint_5_angle_limit
    :initarg :joint_5_angle_limit
    :type cl:boolean
    :initform cl:nil)
   (joint_6_angle_limit
    :reader joint_6_angle_limit
    :initarg :joint_6_angle_limit
    :type cl:boolean
    :initform cl:nil)
   (communication_status_joint_1
    :reader communication_status_joint_1
    :initarg :communication_status_joint_1
    :type cl:boolean
    :initform cl:nil)
   (communication_status_joint_2
    :reader communication_status_joint_2
    :initarg :communication_status_joint_2
    :type cl:boolean
    :initform cl:nil)
   (communication_status_joint_3
    :reader communication_status_joint_3
    :initarg :communication_status_joint_3
    :type cl:boolean
    :initform cl:nil)
   (communication_status_joint_4
    :reader communication_status_joint_4
    :initarg :communication_status_joint_4
    :type cl:boolean
    :initform cl:nil)
   (communication_status_joint_5
    :reader communication_status_joint_5
    :initarg :communication_status_joint_5
    :type cl:boolean
    :initform cl:nil)
   (communication_status_joint_6
    :reader communication_status_joint_6
    :initarg :communication_status_joint_6
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass PiperStatusMsg (<PiperStatusMsg>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PiperStatusMsg>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PiperStatusMsg)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name piper_msgs-msg:<PiperStatusMsg> is deprecated: use piper_msgs-msg:PiperStatusMsg instead.")))

(cl:ensure-generic-function 'ctrl_mode-val :lambda-list '(m))
(cl:defmethod ctrl_mode-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:ctrl_mode-val is deprecated.  Use piper_msgs-msg:ctrl_mode instead.")
  (ctrl_mode m))

(cl:ensure-generic-function 'arm_status-val :lambda-list '(m))
(cl:defmethod arm_status-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:arm_status-val is deprecated.  Use piper_msgs-msg:arm_status instead.")
  (arm_status m))

(cl:ensure-generic-function 'mode_feedback-val :lambda-list '(m))
(cl:defmethod mode_feedback-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:mode_feedback-val is deprecated.  Use piper_msgs-msg:mode_feedback instead.")
  (mode_feedback m))

(cl:ensure-generic-function 'teach_status-val :lambda-list '(m))
(cl:defmethod teach_status-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:teach_status-val is deprecated.  Use piper_msgs-msg:teach_status instead.")
  (teach_status m))

(cl:ensure-generic-function 'motion_status-val :lambda-list '(m))
(cl:defmethod motion_status-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:motion_status-val is deprecated.  Use piper_msgs-msg:motion_status instead.")
  (motion_status m))

(cl:ensure-generic-function 'trajectory_num-val :lambda-list '(m))
(cl:defmethod trajectory_num-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:trajectory_num-val is deprecated.  Use piper_msgs-msg:trajectory_num instead.")
  (trajectory_num m))

(cl:ensure-generic-function 'err_code-val :lambda-list '(m))
(cl:defmethod err_code-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:err_code-val is deprecated.  Use piper_msgs-msg:err_code instead.")
  (err_code m))

(cl:ensure-generic-function 'joint_1_angle_limit-val :lambda-list '(m))
(cl:defmethod joint_1_angle_limit-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:joint_1_angle_limit-val is deprecated.  Use piper_msgs-msg:joint_1_angle_limit instead.")
  (joint_1_angle_limit m))

(cl:ensure-generic-function 'joint_2_angle_limit-val :lambda-list '(m))
(cl:defmethod joint_2_angle_limit-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:joint_2_angle_limit-val is deprecated.  Use piper_msgs-msg:joint_2_angle_limit instead.")
  (joint_2_angle_limit m))

(cl:ensure-generic-function 'joint_3_angle_limit-val :lambda-list '(m))
(cl:defmethod joint_3_angle_limit-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:joint_3_angle_limit-val is deprecated.  Use piper_msgs-msg:joint_3_angle_limit instead.")
  (joint_3_angle_limit m))

(cl:ensure-generic-function 'joint_4_angle_limit-val :lambda-list '(m))
(cl:defmethod joint_4_angle_limit-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:joint_4_angle_limit-val is deprecated.  Use piper_msgs-msg:joint_4_angle_limit instead.")
  (joint_4_angle_limit m))

(cl:ensure-generic-function 'joint_5_angle_limit-val :lambda-list '(m))
(cl:defmethod joint_5_angle_limit-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:joint_5_angle_limit-val is deprecated.  Use piper_msgs-msg:joint_5_angle_limit instead.")
  (joint_5_angle_limit m))

(cl:ensure-generic-function 'joint_6_angle_limit-val :lambda-list '(m))
(cl:defmethod joint_6_angle_limit-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:joint_6_angle_limit-val is deprecated.  Use piper_msgs-msg:joint_6_angle_limit instead.")
  (joint_6_angle_limit m))

(cl:ensure-generic-function 'communication_status_joint_1-val :lambda-list '(m))
(cl:defmethod communication_status_joint_1-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:communication_status_joint_1-val is deprecated.  Use piper_msgs-msg:communication_status_joint_1 instead.")
  (communication_status_joint_1 m))

(cl:ensure-generic-function 'communication_status_joint_2-val :lambda-list '(m))
(cl:defmethod communication_status_joint_2-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:communication_status_joint_2-val is deprecated.  Use piper_msgs-msg:communication_status_joint_2 instead.")
  (communication_status_joint_2 m))

(cl:ensure-generic-function 'communication_status_joint_3-val :lambda-list '(m))
(cl:defmethod communication_status_joint_3-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:communication_status_joint_3-val is deprecated.  Use piper_msgs-msg:communication_status_joint_3 instead.")
  (communication_status_joint_3 m))

(cl:ensure-generic-function 'communication_status_joint_4-val :lambda-list '(m))
(cl:defmethod communication_status_joint_4-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:communication_status_joint_4-val is deprecated.  Use piper_msgs-msg:communication_status_joint_4 instead.")
  (communication_status_joint_4 m))

(cl:ensure-generic-function 'communication_status_joint_5-val :lambda-list '(m))
(cl:defmethod communication_status_joint_5-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:communication_status_joint_5-val is deprecated.  Use piper_msgs-msg:communication_status_joint_5 instead.")
  (communication_status_joint_5 m))

(cl:ensure-generic-function 'communication_status_joint_6-val :lambda-list '(m))
(cl:defmethod communication_status_joint_6-val ((m <PiperStatusMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader piper_msgs-msg:communication_status_joint_6-val is deprecated.  Use piper_msgs-msg:communication_status_joint_6 instead.")
  (communication_status_joint_6 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PiperStatusMsg>) ostream)
  "Serializes a message object of type '<PiperStatusMsg>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'ctrl_mode)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'arm_status)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode_feedback)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'teach_status)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'motion_status)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'trajectory_num)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'err_code)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'joint_1_angle_limit) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'joint_2_angle_limit) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'joint_3_angle_limit) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'joint_4_angle_limit) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'joint_5_angle_limit) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'joint_6_angle_limit) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'communication_status_joint_1) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'communication_status_joint_2) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'communication_status_joint_3) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'communication_status_joint_4) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'communication_status_joint_5) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'communication_status_joint_6) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PiperStatusMsg>) istream)
  "Deserializes a message object of type '<PiperStatusMsg>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'ctrl_mode)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'arm_status)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode_feedback)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'teach_status)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'motion_status)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'trajectory_num)) (cl:read-byte istream))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'err_code) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
    (cl:setf (cl:slot-value msg 'joint_1_angle_limit) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'joint_2_angle_limit) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'joint_3_angle_limit) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'joint_4_angle_limit) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'joint_5_angle_limit) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'joint_6_angle_limit) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'communication_status_joint_1) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'communication_status_joint_2) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'communication_status_joint_3) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'communication_status_joint_4) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'communication_status_joint_5) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'communication_status_joint_6) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PiperStatusMsg>)))
  "Returns string type for a message object of type '<PiperStatusMsg>"
  "piper_msgs/PiperStatusMsg")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PiperStatusMsg)))
  "Returns string type for a message object of type 'PiperStatusMsg"
  "piper_msgs/PiperStatusMsg")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PiperStatusMsg>)))
  "Returns md5sum for a message object of type '<PiperStatusMsg>"
  "18e0aaa943372aaa58f1495907dd9a17")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PiperStatusMsg)))
  "Returns md5sum for a message object of type 'PiperStatusMsg"
  "18e0aaa943372aaa58f1495907dd9a17")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PiperStatusMsg>)))
  "Returns full string definition for message of type '<PiperStatusMsg>"
  (cl:format cl:nil "uint8 ctrl_mode~%uint8 arm_status~%uint8 mode_feedback~%uint8 teach_status~%uint8 motion_status~%uint8 trajectory_num~%int64 err_code~%bool joint_1_angle_limit~%bool joint_2_angle_limit~%bool joint_3_angle_limit~%bool joint_4_angle_limit~%bool joint_5_angle_limit~%bool joint_6_angle_limit~%bool communication_status_joint_1~%bool communication_status_joint_2~%bool communication_status_joint_3~%bool communication_status_joint_4~%bool communication_status_joint_5~%bool communication_status_joint_6~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PiperStatusMsg)))
  "Returns full string definition for message of type 'PiperStatusMsg"
  (cl:format cl:nil "uint8 ctrl_mode~%uint8 arm_status~%uint8 mode_feedback~%uint8 teach_status~%uint8 motion_status~%uint8 trajectory_num~%int64 err_code~%bool joint_1_angle_limit~%bool joint_2_angle_limit~%bool joint_3_angle_limit~%bool joint_4_angle_limit~%bool joint_5_angle_limit~%bool joint_6_angle_limit~%bool communication_status_joint_1~%bool communication_status_joint_2~%bool communication_status_joint_3~%bool communication_status_joint_4~%bool communication_status_joint_5~%bool communication_status_joint_6~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PiperStatusMsg>))
  (cl:+ 0
     1
     1
     1
     1
     1
     1
     8
     1
     1
     1
     1
     1
     1
     1
     1
     1
     1
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PiperStatusMsg>))
  "Converts a ROS message object to a list"
  (cl:list 'PiperStatusMsg
    (cl:cons ':ctrl_mode (ctrl_mode msg))
    (cl:cons ':arm_status (arm_status msg))
    (cl:cons ':mode_feedback (mode_feedback msg))
    (cl:cons ':teach_status (teach_status msg))
    (cl:cons ':motion_status (motion_status msg))
    (cl:cons ':trajectory_num (trajectory_num msg))
    (cl:cons ':err_code (err_code msg))
    (cl:cons ':joint_1_angle_limit (joint_1_angle_limit msg))
    (cl:cons ':joint_2_angle_limit (joint_2_angle_limit msg))
    (cl:cons ':joint_3_angle_limit (joint_3_angle_limit msg))
    (cl:cons ':joint_4_angle_limit (joint_4_angle_limit msg))
    (cl:cons ':joint_5_angle_limit (joint_5_angle_limit msg))
    (cl:cons ':joint_6_angle_limit (joint_6_angle_limit msg))
    (cl:cons ':communication_status_joint_1 (communication_status_joint_1 msg))
    (cl:cons ':communication_status_joint_2 (communication_status_joint_2 msg))
    (cl:cons ':communication_status_joint_3 (communication_status_joint_3 msg))
    (cl:cons ':communication_status_joint_4 (communication_status_joint_4 msg))
    (cl:cons ':communication_status_joint_5 (communication_status_joint_5 msg))
    (cl:cons ':communication_status_joint_6 (communication_status_joint_6 msg))
))
