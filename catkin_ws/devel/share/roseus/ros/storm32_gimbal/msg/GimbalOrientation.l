;; Auto-generated. Do not edit!


(when (boundp 'storm32_gimbal::GimbalOrientation)
  (if (not (find-package "STORM32_GIMBAL"))
    (make-package "STORM32_GIMBAL"))
  (shadow 'GimbalOrientation (find-package "STORM32_GIMBAL")))
(unless (find-package "STORM32_GIMBAL::GIMBALORIENTATION")
  (make-package "STORM32_GIMBAL::GIMBALORIENTATION"))

(in-package "ROS")
;;//! \htmlinclude GimbalOrientation.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass storm32_gimbal::GimbalOrientation
  :super ros::object
  :slots (_orientation _unlimited ))

(defmethod storm32_gimbal::GimbalOrientation
  (:init
   (&key
    ((:orientation __orientation) (instance geometry_msgs::Quaternion :init))
    ((:unlimited __unlimited) nil)
    )
   (send-super :init)
   (setq _orientation __orientation)
   (setq _unlimited __unlimited)
   self)
  (:orientation
   (&rest __orientation)
   (if (keywordp (car __orientation))
       (send* _orientation __orientation)
     (progn
       (if __orientation (setq _orientation (car __orientation)))
       _orientation)))
  (:unlimited
   (&optional (__unlimited :null))
   (if (not (eq __unlimited :null)) (setq _unlimited __unlimited)) _unlimited)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Quaternion _orientation
    (send _orientation :serialization-length)
    ;; bool _unlimited
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Quaternion _orientation
       (send _orientation :serialize s)
     ;; bool _unlimited
       (if _unlimited (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Quaternion _orientation
     (send _orientation :deserialize buf ptr-) (incf ptr- (send _orientation :serialization-length))
   ;; bool _unlimited
     (setq _unlimited (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get storm32_gimbal::GimbalOrientation :md5sum-) "429b834c7e4456a2d8d33b9a6ac43342")
(setf (get storm32_gimbal::GimbalOrientation :datatype-) "storm32_gimbal/GimbalOrientation")
(setf (get storm32_gimbal::GimbalOrientation :definition-)
      "geometry_msgs/Quaternion orientation
bool unlimited

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

")



(provide :storm32_gimbal/GimbalOrientation "429b834c7e4456a2d8d33b9a6ac43342")


