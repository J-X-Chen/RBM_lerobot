;; Auto-generated. Do not edit!


(when (boundp 'piper_msgs::PiperStatusMsg)
  (if (not (find-package "PIPER_MSGS"))
    (make-package "PIPER_MSGS"))
  (shadow 'PiperStatusMsg (find-package "PIPER_MSGS")))
(unless (find-package "PIPER_MSGS::PIPERSTATUSMSG")
  (make-package "PIPER_MSGS::PIPERSTATUSMSG"))

(in-package "ROS")
;;//! \htmlinclude PiperStatusMsg.msg.html


(defclass piper_msgs::PiperStatusMsg
  :super ros::object
  :slots (_ctrl_mode _arm_status _mode_feedback _teach_status _motion_status _trajectory_num _err_code _joint_1_angle_limit _joint_2_angle_limit _joint_3_angle_limit _joint_4_angle_limit _joint_5_angle_limit _joint_6_angle_limit _communication_status_joint_1 _communication_status_joint_2 _communication_status_joint_3 _communication_status_joint_4 _communication_status_joint_5 _communication_status_joint_6 ))

(defmethod piper_msgs::PiperStatusMsg
  (:init
   (&key
    ((:ctrl_mode __ctrl_mode) 0)
    ((:arm_status __arm_status) 0)
    ((:mode_feedback __mode_feedback) 0)
    ((:teach_status __teach_status) 0)
    ((:motion_status __motion_status) 0)
    ((:trajectory_num __trajectory_num) 0)
    ((:err_code __err_code) 0)
    ((:joint_1_angle_limit __joint_1_angle_limit) nil)
    ((:joint_2_angle_limit __joint_2_angle_limit) nil)
    ((:joint_3_angle_limit __joint_3_angle_limit) nil)
    ((:joint_4_angle_limit __joint_4_angle_limit) nil)
    ((:joint_5_angle_limit __joint_5_angle_limit) nil)
    ((:joint_6_angle_limit __joint_6_angle_limit) nil)
    ((:communication_status_joint_1 __communication_status_joint_1) nil)
    ((:communication_status_joint_2 __communication_status_joint_2) nil)
    ((:communication_status_joint_3 __communication_status_joint_3) nil)
    ((:communication_status_joint_4 __communication_status_joint_4) nil)
    ((:communication_status_joint_5 __communication_status_joint_5) nil)
    ((:communication_status_joint_6 __communication_status_joint_6) nil)
    )
   (send-super :init)
   (setq _ctrl_mode (round __ctrl_mode))
   (setq _arm_status (round __arm_status))
   (setq _mode_feedback (round __mode_feedback))
   (setq _teach_status (round __teach_status))
   (setq _motion_status (round __motion_status))
   (setq _trajectory_num (round __trajectory_num))
   (setq _err_code (round __err_code))
   (setq _joint_1_angle_limit __joint_1_angle_limit)
   (setq _joint_2_angle_limit __joint_2_angle_limit)
   (setq _joint_3_angle_limit __joint_3_angle_limit)
   (setq _joint_4_angle_limit __joint_4_angle_limit)
   (setq _joint_5_angle_limit __joint_5_angle_limit)
   (setq _joint_6_angle_limit __joint_6_angle_limit)
   (setq _communication_status_joint_1 __communication_status_joint_1)
   (setq _communication_status_joint_2 __communication_status_joint_2)
   (setq _communication_status_joint_3 __communication_status_joint_3)
   (setq _communication_status_joint_4 __communication_status_joint_4)
   (setq _communication_status_joint_5 __communication_status_joint_5)
   (setq _communication_status_joint_6 __communication_status_joint_6)
   self)
  (:ctrl_mode
   (&optional __ctrl_mode)
   (if __ctrl_mode (setq _ctrl_mode __ctrl_mode)) _ctrl_mode)
  (:arm_status
   (&optional __arm_status)
   (if __arm_status (setq _arm_status __arm_status)) _arm_status)
  (:mode_feedback
   (&optional __mode_feedback)
   (if __mode_feedback (setq _mode_feedback __mode_feedback)) _mode_feedback)
  (:teach_status
   (&optional __teach_status)
   (if __teach_status (setq _teach_status __teach_status)) _teach_status)
  (:motion_status
   (&optional __motion_status)
   (if __motion_status (setq _motion_status __motion_status)) _motion_status)
  (:trajectory_num
   (&optional __trajectory_num)
   (if __trajectory_num (setq _trajectory_num __trajectory_num)) _trajectory_num)
  (:err_code
   (&optional __err_code)
   (if __err_code (setq _err_code __err_code)) _err_code)
  (:joint_1_angle_limit
   (&optional (__joint_1_angle_limit :null))
   (if (not (eq __joint_1_angle_limit :null)) (setq _joint_1_angle_limit __joint_1_angle_limit)) _joint_1_angle_limit)
  (:joint_2_angle_limit
   (&optional (__joint_2_angle_limit :null))
   (if (not (eq __joint_2_angle_limit :null)) (setq _joint_2_angle_limit __joint_2_angle_limit)) _joint_2_angle_limit)
  (:joint_3_angle_limit
   (&optional (__joint_3_angle_limit :null))
   (if (not (eq __joint_3_angle_limit :null)) (setq _joint_3_angle_limit __joint_3_angle_limit)) _joint_3_angle_limit)
  (:joint_4_angle_limit
   (&optional (__joint_4_angle_limit :null))
   (if (not (eq __joint_4_angle_limit :null)) (setq _joint_4_angle_limit __joint_4_angle_limit)) _joint_4_angle_limit)
  (:joint_5_angle_limit
   (&optional (__joint_5_angle_limit :null))
   (if (not (eq __joint_5_angle_limit :null)) (setq _joint_5_angle_limit __joint_5_angle_limit)) _joint_5_angle_limit)
  (:joint_6_angle_limit
   (&optional (__joint_6_angle_limit :null))
   (if (not (eq __joint_6_angle_limit :null)) (setq _joint_6_angle_limit __joint_6_angle_limit)) _joint_6_angle_limit)
  (:communication_status_joint_1
   (&optional (__communication_status_joint_1 :null))
   (if (not (eq __communication_status_joint_1 :null)) (setq _communication_status_joint_1 __communication_status_joint_1)) _communication_status_joint_1)
  (:communication_status_joint_2
   (&optional (__communication_status_joint_2 :null))
   (if (not (eq __communication_status_joint_2 :null)) (setq _communication_status_joint_2 __communication_status_joint_2)) _communication_status_joint_2)
  (:communication_status_joint_3
   (&optional (__communication_status_joint_3 :null))
   (if (not (eq __communication_status_joint_3 :null)) (setq _communication_status_joint_3 __communication_status_joint_3)) _communication_status_joint_3)
  (:communication_status_joint_4
   (&optional (__communication_status_joint_4 :null))
   (if (not (eq __communication_status_joint_4 :null)) (setq _communication_status_joint_4 __communication_status_joint_4)) _communication_status_joint_4)
  (:communication_status_joint_5
   (&optional (__communication_status_joint_5 :null))
   (if (not (eq __communication_status_joint_5 :null)) (setq _communication_status_joint_5 __communication_status_joint_5)) _communication_status_joint_5)
  (:communication_status_joint_6
   (&optional (__communication_status_joint_6 :null))
   (if (not (eq __communication_status_joint_6 :null)) (setq _communication_status_joint_6 __communication_status_joint_6)) _communication_status_joint_6)
  (:serialization-length
   ()
   (+
    ;; uint8 _ctrl_mode
    1
    ;; uint8 _arm_status
    1
    ;; uint8 _mode_feedback
    1
    ;; uint8 _teach_status
    1
    ;; uint8 _motion_status
    1
    ;; uint8 _trajectory_num
    1
    ;; int64 _err_code
    8
    ;; bool _joint_1_angle_limit
    1
    ;; bool _joint_2_angle_limit
    1
    ;; bool _joint_3_angle_limit
    1
    ;; bool _joint_4_angle_limit
    1
    ;; bool _joint_5_angle_limit
    1
    ;; bool _joint_6_angle_limit
    1
    ;; bool _communication_status_joint_1
    1
    ;; bool _communication_status_joint_2
    1
    ;; bool _communication_status_joint_3
    1
    ;; bool _communication_status_joint_4
    1
    ;; bool _communication_status_joint_5
    1
    ;; bool _communication_status_joint_6
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _ctrl_mode
       (write-byte _ctrl_mode s)
     ;; uint8 _arm_status
       (write-byte _arm_status s)
     ;; uint8 _mode_feedback
       (write-byte _mode_feedback s)
     ;; uint8 _teach_status
       (write-byte _teach_status s)
     ;; uint8 _motion_status
       (write-byte _motion_status s)
     ;; uint8 _trajectory_num
       (write-byte _trajectory_num s)
     ;; int64 _err_code
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _err_code (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _err_code) (= (length (_err_code . bv)) 2)) ;; bignum
              (write-long (ash (elt (_err_code . bv) 0) 0) s)
              (write-long (ash (elt (_err_code . bv) 1) -1) s))
             ((and (class _err_code) (= (length (_err_code . bv)) 1)) ;; big1
              (write-long (elt (_err_code . bv) 0) s)
              (write-long (if (>= _err_code 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _err_code s)(write-long (if (>= _err_code 0) 0 #xffffffff) s)))
     ;; bool _joint_1_angle_limit
       (if _joint_1_angle_limit (write-byte -1 s) (write-byte 0 s))
     ;; bool _joint_2_angle_limit
       (if _joint_2_angle_limit (write-byte -1 s) (write-byte 0 s))
     ;; bool _joint_3_angle_limit
       (if _joint_3_angle_limit (write-byte -1 s) (write-byte 0 s))
     ;; bool _joint_4_angle_limit
       (if _joint_4_angle_limit (write-byte -1 s) (write-byte 0 s))
     ;; bool _joint_5_angle_limit
       (if _joint_5_angle_limit (write-byte -1 s) (write-byte 0 s))
     ;; bool _joint_6_angle_limit
       (if _joint_6_angle_limit (write-byte -1 s) (write-byte 0 s))
     ;; bool _communication_status_joint_1
       (if _communication_status_joint_1 (write-byte -1 s) (write-byte 0 s))
     ;; bool _communication_status_joint_2
       (if _communication_status_joint_2 (write-byte -1 s) (write-byte 0 s))
     ;; bool _communication_status_joint_3
       (if _communication_status_joint_3 (write-byte -1 s) (write-byte 0 s))
     ;; bool _communication_status_joint_4
       (if _communication_status_joint_4 (write-byte -1 s) (write-byte 0 s))
     ;; bool _communication_status_joint_5
       (if _communication_status_joint_5 (write-byte -1 s) (write-byte 0 s))
     ;; bool _communication_status_joint_6
       (if _communication_status_joint_6 (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _ctrl_mode
     (setq _ctrl_mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _arm_status
     (setq _arm_status (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _mode_feedback
     (setq _mode_feedback (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _teach_status
     (setq _teach_status (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _motion_status
     (setq _motion_status (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _trajectory_num
     (setq _trajectory_num (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; int64 _err_code
#+(or :alpha :irix6 :x86_64)
      (setf _err_code (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _err_code (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; bool _joint_1_angle_limit
     (setq _joint_1_angle_limit (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _joint_2_angle_limit
     (setq _joint_2_angle_limit (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _joint_3_angle_limit
     (setq _joint_3_angle_limit (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _joint_4_angle_limit
     (setq _joint_4_angle_limit (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _joint_5_angle_limit
     (setq _joint_5_angle_limit (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _joint_6_angle_limit
     (setq _joint_6_angle_limit (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _communication_status_joint_1
     (setq _communication_status_joint_1 (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _communication_status_joint_2
     (setq _communication_status_joint_2 (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _communication_status_joint_3
     (setq _communication_status_joint_3 (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _communication_status_joint_4
     (setq _communication_status_joint_4 (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _communication_status_joint_5
     (setq _communication_status_joint_5 (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _communication_status_joint_6
     (setq _communication_status_joint_6 (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get piper_msgs::PiperStatusMsg :md5sum-) "18e0aaa943372aaa58f1495907dd9a17")
(setf (get piper_msgs::PiperStatusMsg :datatype-) "piper_msgs/PiperStatusMsg")
(setf (get piper_msgs::PiperStatusMsg :definition-)
      "uint8 ctrl_mode
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


")



(provide :piper_msgs/PiperStatusMsg "18e0aaa943372aaa58f1495907dd9a17")


