# This message describes a notification of a change in the objects server.

# Notification type.
uint8 ADDED_OBJECT=0
uint8 UPDATED_OBJECT=1
uint8 DELETED_OBJECT=2
uint8 SET_IMAGE=3
uint8 SET_COMPRESSED_IMAGE=4
uint8 DELETED_IMAGE=5
uint8 RELOAD_ALL=6
uint8 CLEAR_ALL=7
uint8 type

# Associated object ID.
# Set unless type is RELOAD_ALL or CLEAR_ALL.
uint64 id

# Associated object definition.
# Set if type is ADDED_OBJECT or UPDATED_OBJECT.
Object object

# Associated object thumbnail.
sensor_msgs/Image image   # Set if SET_IMAGE.
sensor_msgs/CompressedImage compressed_image  # Set if SET_IMAGE_COMPRESSED.
