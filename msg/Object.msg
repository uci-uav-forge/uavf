# This message describes an object.
# This message definition is to be shared for all object types.

# Object type.
ObjectType type

# Object latitude and longitude (optional).
# Need only be set for standard, QRC and emergent objects.
float64 latitude
float64 longitude

# Object orientation (optional).
# Need only be set for standard and off-axis objects.
Orientation orientation

# Object shape (optional).
# Need only be set for standard and off-axis objects.
Shape shape

# Object background and foreground colors (optional).
# Need only be set for standard and off-axis objects.
Color background_color
Color alphanumeric_color

# Object alphanumeric message (optional).
# Need only be set for standard and off-axis objects.
string alphanumeric

# Object description (optional).
# For QRC object: exact QRC message.
# For emergent object: general description of the emergent object.
# Need only be set for QRC and objects.
string description

# Whether this object was detected, localized, and characterized autonomously,
# -- with no human input --.
# No more than six objects should be marked as autonomous as per the rules.
# Need only be set for standard objects.
bool autonomous
