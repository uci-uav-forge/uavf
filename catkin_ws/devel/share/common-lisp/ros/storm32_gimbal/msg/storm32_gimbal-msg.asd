
(cl:in-package :asdf)

(defsystem "storm32_gimbal-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "GimbalOrientation" :depends-on ("_package_GimbalOrientation"))
    (:file "_package_GimbalOrientation" :depends-on ("_package"))
  ))