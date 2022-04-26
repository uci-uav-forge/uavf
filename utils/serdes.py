#!/usr/bin/env python3
from geographic_msgs.msg import GeoPointStamped, GeoPoint
from std_msgs.msg import Header, Time
from interop.msg import (Color, FlyZone, FlyZoneArray, Orientation, Shape,
                         Object, ObjectType, GeoCylinder, GeoPolygonStamped,
                         GeoCylinderArrayStamped, WayPoints)

from proto import interop_api_pb2

class MissionDeserializer:
	
	def get_dict(self, mission_data):
		mission_data_dict = {
		"lost_comms_pos": self.format_lost_comm_pos(mission_data.lost_comms_pos),
		"flyzones": self.format_flyzones(mission_data.fly_zones),
		"waypoints": self.format_waypoints(mission_data.waypoints),
		"search_grid_points": self.format_search_grid_points(mission_data.search_grid_points),
		"off_axis_odlc_pos": self.format_off_axis_odlc_pos(mission_data.off_axis_odlc_pos),
		"emergent_last_known_pos": self.format_emergent_last_known_pos(mission_data.emergent_last_known_pos),
		"air_drop_boundary_points": self.format_air_drop_boundary_points(mission_data.air_drop_boundary_points),
		"air_drop_pos": self.format_air_drop_pos(mission_data.air_drop_pos),
		"ugv_drive_pos": self.format_ugv_drive_pos(mission_data.ugv_drive_pos),
		"map_center_pos": self.format_map_center_pos(mission_data.map_center_pos),
		"stationary_obstacles": self.format_stationary_obstacles(mission_data.stationary_obstacles)
		}

		return mission_data_dict


	def format_lost_comm_pos(self, data):
		pass

	def format_flyzones(self, data):
		pass

	def format_waypoints(self, data):
		pass

	def format_search_grid_points(self, data):
		pass

	def format_off_axis_odlc_pos(self, data):
		pass

	def format_emergent_last_known_pos(self, data):
		pass

	def format_air_drop_boundary_points(self, data):
		pass

	def format_air_drop_pos(self, data):
		pass

	def format_ugv_drive_pos(self, data):
		pass

	def format_map_center_pos(self, data):
		pass

	def format_stationary_obstacles(self, data):
		pass

class TelemetrySerializer:
	
	def format(self, telemetry_data):
		pass

class OdlcSerializer:
	
	def format(self, odlc_data):
		pass

class MapSerializer:

	def format(self, map_data):
		pass