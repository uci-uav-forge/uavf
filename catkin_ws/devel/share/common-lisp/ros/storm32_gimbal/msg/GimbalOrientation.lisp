; Auto-generated. Do not edit!


(cl:in-package storm32_gimbal-msg)


;//! \htmlinclude GimbalOrientation.msg.html

(cl:defclass <GimbalOrientation> (roslisp-msg-protocol:ros-message)
  ((orientation
    :reader orientation
    :initarg :orientation
    :type geometry_msgs-msg:Quaternion
    :initform (cl:make-instance 'geometry_msgs-msg:Quaternion))
   (unlimited
    :reader unlimited
    :initarg :unlimited
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass GimbalOrientation (<GimbalOrientation>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GimbalOrientation>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GimbalOrientation)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name storm32_gimbal-msg:<GimbalOrientation> is deprecated: use storm32_gimbal-msg:GimbalOrientation instead.")))

(cl:ensure-generic-function 'orientation-val :lambda-list '(m))
(cl:defmethod orientation-val ((m <GimbalOrientation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader storm32_gimbal-msg:orientation-val is deprecated.  Use storm32_gimbal-msg:orientation instead.")
  (orientation m))

(cl:ensure-generic-function 'unlimited-val :lambda-list '(m))
(cl:defmethod unlimited-val ((m <GimbalOrientation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader storm32_gimbal-msg:unlimited-val is deprecated.  Use storm32_gimbal-msg:unlimited instead.")
  (unlimited m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GimbalOrientation>) ostream)
  "Serializes a message object of type '<GimbalOrientation>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'orientation) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'unlimited) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GimbalOrientation>) istream)
  "Deserializes a message object of type '<GimbalOrientation>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'orientation) istream)
    (cl:setf (cl:slot-value msg 'unlimited) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GimbalOrientation>)))
  "Returns string type for a message object of type '<GimbalOrientation>"
  "storm32_gimbal/GimbalOrientation")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GimbalOrientation)))
  "Returns string type for a message object of type 'GimbalOrientation"
  "storm32_gimbal/GimbalOrientation")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GimbalOrientation>)))
  "Returns md5sum for a message object of type '<GimbalOrientation>"
  "429b834c7e4456a2d8d33b9a6ac43342")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GimbalOrientation)))
  "Returns md5sum for a message object of type 'GimbalOrientation"
  "429b834c7e4456a2d8d33b9a6ac43342")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GimbalOrientation>)))
  "Returns full string definition for message of type '<GimbalOrientation>"
  (cl:format cl:nil "geometry_msgs/Quaternion orientation~%bool unlimited~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GimbalOrientation)))
  "Returns full string definition for message of type 'GimbalOrientation"
  (cl:format cl:nil "geometry_msgs/Quaternion orientation~%bool unlimited~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GimbalOrientation>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'orientation))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GimbalOrientation>))
  "Converts a ROS message object to a list"
  (cl:list 'GimbalOrientation
    (cl:cons ':orientation (orientation msg))
    (cl:cons ':unlimited (unlimited msg))
))
