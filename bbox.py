#!/usr/bin/env python

import logging
import os
import threading
import pickle
import string
import random
import time
from datetime import datetime
from pathlib import Path
from enum import Enum
import argparse
import traceback
import weakref
from dataclasses import dataclass, fields
import json

from typing import NamedTuple
from typing import List, Tuple, Optional, Any

import numpy as np

import imageio

import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w

if 'CARLAPATH' in os.environ:
    import sys
    sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.11-py3.7-linux-x86_64.egg'))
    sys.path.append(os.environ['CARLAPATH'])

import carla
from carla import ColorConverter as cc

import carla_vehicle_annotator

mutex = threading.Lock()

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

CAMERA_ATTRIBUTES = {
    "image_size_x": str(VIEW_WIDTH),
    "image_size_y": str(VIEW_HEIGHT),
    "fov": str(VIEW_FOV),
}

CAMERA_BP_STR = "sensor.camera.rgb"
SEMANTIC_BP_STR = "sensor.camera.semantic_segmentation"
DEPTH_BP_STR = "sensor.camera.depth"
LIDAR_BP_STR = "sensor.lidar.ray_cast_semantic"

COLOR_CONVERTERS = {
    CAMERA_BP_STR: cc.Raw,
    SEMANTIC_BP_STR: cc.CityScapesPalette,
    DEPTH_BP_STR: cc.Depth,
}

PROCESS_EVERY_N_FRAMES = 10


class SensorLocation(Enum):
    FC = 1
    FL = 2
    FR = 3


SENSOR_POSITIONS = {SensorLocation.FC: 0, SensorLocation.FL: -0.483, SensorLocation.FR: 0.483}


class Box(NamedTuple):
    points: List[List[Tuple[float, float]]]
    distance_to_camera: float
    max_distance_to_center: float
    vehicle: carla.Vehicle

    def without_vehicle(self):
        return Box(self.points, self.distance_to_camera, self.max_distance_to_center, None)


@dataclass
class CameraLidarPair:
    camera_image: Optional[Any] = None
    lidar_data: Optional[Any] = None


@dataclass
class WorldBoxInfo:
    # Contains info needed for bounding boxes generation.

    boxes_FC: Optional[List[Box]] = None
    boxes_FL: Optional[List[Box]] = None
    boxes_FR: Optional[List[Box]] = None

    transform: Optional[carla.Transform] = None
    velocity: Optional[Any] = None
    angular_velocity: Optional[Any] = None
    acceleration: Optional[Any] = None

    camera_image_FC: Optional[Any] = None
    lidar_data_FC: Optional[Any] = None

    camera_image_FL: Optional[Any] = None
    lidar_data_FL: Optional[Any] = None

    camera_image_FR: Optional[Any] = None
    lidar_data_FR: Optional[Any] = None

    is_at_traffic_light: Optional[bool] = None

    def is_complete(self):
        # checks if the info is complete to build the bounding box.

        for f in fields(self):
            if getattr(self, f.name) is None:
                return False

        return True


class ClientSideBoundingBoxes(object):
    @staticmethod
    def check_bboxes_consistency_lidar(bboxes, camera, lidar_data, max_dist, min_detect):
        filtered_data = carla_vehicle_annotator.filter_lidar(lidar_data, camera, max_dist)
        filtered_data = np.array([p for p in filtered_data if p.object_idx != 0])
        vehicles = [b.vehicle for b in bboxes]
        filtered_data = carla_vehicle_annotator.get_points_id(filtered_data, vehicles, camera, max_dist)

        visible_id, idx_counts = np.unique([p.object_idx for p in filtered_data], return_counts=True)

        visible_vehicles = set()
        for v, c in zip(visible_id, idx_counts):
            if c >= min_detect:
                visible_vehicles.add(v)

        res = []
        for b in bboxes:
            if b.vehicle.id in visible_vehicles:
                res.append(3)
            else:
                res.append(0)
        return res

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        bounding_boxes = list(filter(None, bounding_boxes))

        # filter objects behind camera
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(bb_surface, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface.set_colorkey((0, 0, 0))
        color = (0, 255, 0)
        for bbox in bounding_boxes:
            points = bbox['points']
            # draw lines
            # base
            print("drawing points", points)

            pygame.draw.line(bb_surface, color, points[0], points[1])
            pygame.draw.line(bb_surface, color, points[0], points[1])
            pygame.draw.line(bb_surface, color, points[1], points[2])
            pygame.draw.line(bb_surface, color, points[2], points[3])
            pygame.draw.line(bb_surface, color, points[3], points[0])

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)

        # get distance to car
        # get max size from the car.
        # print(box.location)         # Location relative to the vehicle.
        # print(box.extent)           # XYZ half-box extents in meters.
        center = carla.Location(vehicle.get_location() + vehicle.bounding_box.location)
        distance_to_camera = camera.get_location().distance(center)
        max_distance_to_center = center.distance(center + vehicle.bounding_box.extent)

        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)

        if not all(camera_bbox[:, 2] > 0):
            return None

        else:
            max_x = np.max(camera_bbox[:, 0])
            max_y = np.max(camera_bbox[:, 1])
            min_x = np.min(camera_bbox[:, 0])
            min_y = np.min(camera_bbox[:, 1])
            points = [(max_x, max_y), (max_x, min_y), (min_x, min_y), (min_x, max_y)]

            res = Box(points, distance_to_camera, max_distance_to_center, vehicle)

            return res

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(
        self,
        map_id,
        traffic_density,
        session_length,
        destination_dir,
        visualize=True,
    ):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        self.display = None
        self.lidar = None
        self.depth = None
        self.semantic = None

        self.map_id = map_id
        self.traffic_density = traffic_density
        self.session_length = session_length
        self.destination_dir = destination_dir
        self.visualize = visualize

        self.camera_attributes = {
            "image_size_x": str(VIEW_WIDTH),
            "image_size_y": str(VIEW_HEIGHT),
            "fov": str(VIEW_FOV),
        }

        self.data_buffer = dict()

    def blueprint(self, bp_str):
        camera_bp = self.world.get_blueprint_library().find(bp_str)
        # print(bp_str)

        if bp_str == LIDAR_BP_STR:
            camera_bp.set_attribute("channels", "64")
            camera_bp.set_attribute("points_per_second", "1120000")
            camera_bp.set_attribute("upper_fov", "45")
            camera_bp.set_attribute("lower_fov", "-45")
            camera_bp.set_attribute("range", "100")
            camera_bp.set_attribute("rotation_frequency", "20")
        else:
            print("adding stuff")
            for k, v in CAMERA_ATTRIBUTES.items():
                camera_bp.set_attribute(k, v)

        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """Set synchronous mode."""
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def spawn_vehicles(self, density):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = list(filter(lambda x: x.get_attribute("number_of_wheels").as_int() == 4, blueprints))

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points) - 1
        number_of_vehicles = int(density * number_of_spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = "requested %d vehicles, but could only find %d spawn points"
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        actor_list = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                actor_list.append(response.actor_id)

        return actor_list, spawn_points[n:]

    def setup_car(self, spawns):
        """Spawn actor-vehicle to be controled."""
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")[0]
        car_bp.set_attribute("role_name", "autopilot")
        location = random.choice(spawns)
        self.car = self.world.spawn_actor(car_bp, location)
        self.car.set_autopilot(True)

    def setup_sensor(self, bp_str: str, location: SensorLocation):
        """Spawn actor-camera to be used to render view.

        Set calibration for client-side boxes rendering.
        """
        # camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(
            carla.Location(x=0.9, z=1.508, y=SENSOR_POSITIONS[location]), carla.Rotation(pitch=0)
        )
        bp = self.blueprint(bp_str)
        sensor = self.world.spawn_actor(bp, camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)

        def data_lambda(data):
            weak_self().new_data(weak_self, bp_str, location, data)

        sensor.listen(data_lambda)

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        sensor.calibration = calibration
        return sensor

    def control(self, car):
        """Apply control to main car based on pygame pressed keys.

        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1.0, min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1.0, max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def new_data(weak_self, bp_str, location, data):

        self = weak_self()
        if data.frame_number % PROCESS_EVERY_N_FRAMES == 0:
            if data.frame_number not in self.data_buffer:
                self.data_buffer[data.frame_number] = WorldBoxInfo()
            print("got data for ", bp_str, "on time", data.frame_number, "data", data)

            if bp_str == CAMERA_BP_STR:
                if location == SensorLocation.FC:
                    self.data_buffer[data.frame_number].camera_image_FC = data
                elif location == SensorLocation.FL:
                    self.data_buffer[data.frame_number].camera_image_FL = data
                elif location == SensorLocation.FR:
                    self.data_buffer[data.frame_number].camera_image_FR = data

            elif bp_str == LIDAR_BP_STR:
                if location == SensorLocation.FC:
                    self.data_buffer[data.frame_number].lidar_data_FC = data
                elif location == SensorLocation.FL:
                    self.data_buffer[data.frame_number].lidar_data_FL = data
                elif location == SensorLocation.FR:
                    self.data_buffer[data.frame_number].lidar_data_FR = data

    def render(self, display, image_array, boxes):
        """Transform image from camera sensor and blits it to main pygame display."""
        if self.visualize:
            image_array = image_array[:, :, :3]  # remove alpha
            image_array = image_array[:, :, ::-1]  # flip rgb
            surface = pygame.surfarray.make_surface(image_array.swapaxes(0, 1))
            ClientSideBoundingBoxes.draw_bounding_boxes(surface, boxes)
            display.blit(surface, (0, 0))

    def setup_sensors(self):
        """Set up sensors."""
        self.camera_FC = self.setup_sensor(CAMERA_BP_STR, location=SensorLocation.FC)
        self.camera_FL = self.setup_sensor(CAMERA_BP_STR, location=SensorLocation.FL)
        self.camera_FR = self.setup_sensor(CAMERA_BP_STR, location=SensorLocation.FR)

        self.lidar_FC = self.setup_sensor(LIDAR_BP_STR, location=SensorLocation.FC)
        self.lidar_FL = self.setup_sensor(LIDAR_BP_STR, location=SensorLocation.FL)
        self.lidar_FR = self.setup_sensor(LIDAR_BP_STR, location=SensorLocation.FR)

    def game_loop(self):
        """Run main program loop."""
        date_time = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

        self.run_id = "ver1_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10)) + date_time
        print("run id is", self.run_id)

        try:
            if self.visualize:
                pygame.init()

            self.client = carla.Client("127.0.0.1", 2000)
            self.client.set_timeout(60.0)

            self.world = self.client.load_world(f"Town{self.map_id:02d}_Opt")
            self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)

            self.weather = carla.WeatherParameters(
                cloudiness=random.randint(0, 100),
                precipitation=random.randint(0, 1) * random.randint(1, 100),
                sun_altitude_angle=random.randint(-45, 90),
                precipitation_deposits=random.randint(0, 100),
            )
            self.world.set_weather(self.weather)
            self.current_run_dir = (Path(self.destination_dir) / self.run_id)

            self.current_run_dir.mkdir(parents=True, exist_ok=True)
            with (self.current_run_dir /  "session_info.json").open("w") as f:
                d = {
                    "town": self.map_id,
                    "traffic_density": self.traffic_density,
                    "cloudiness": self.weather.cloudiness,
                    "precipitation": self.weather.precipitation,
                    "sun_altitude_angle": self.weather.sun_altitude_angle,
                    "precipitation_deposits": self.weather.precipitation_deposits,
                }
                json.dump(d, f)
                print(json.dumps(d))

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)

            self.tm = self.client.get_trafficmanager()
            self.tm.set_synchronous_mode(True)

            actors, spawns = self.spawn_vehicles(self.traffic_density)
            self.setup_car(spawns)

            self.setup_sensors()

            if self.visualize:
                self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter("vehicle.*")
            self.vehicles = vehicles

            if self.weather.sun_altitude_angle < 10:
                for v in self.vehicles:
                    v.set_light_state(
                        carla.VehicleLightState(carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position)
                    )

            def on_tick(timestamp):
                with mutex:
                    print("frame", timestamp.frame)
                    if timestamp.frame % PROCESS_EVERY_N_FRAMES == 0:
                        print("processing")

                        if timestamp.frame_count not in self.data_buffer:
                            self.data_buffer[timestamp.frame_count] = WorldBoxInfo()

                        self.data_buffer[timestamp.frame_count].boxes_FC = ClientSideBoundingBoxes.get_bounding_boxes(
                            vehicles, self.camera_FC
                        )
                        self.data_buffer[timestamp.frame_count].boxes_FL = ClientSideBoundingBoxes.get_bounding_boxes(
                            vehicles, self.camera_FL
                        )
                        self.data_buffer[timestamp.frame_count].boxes_FR = ClientSideBoundingBoxes.get_bounding_boxes(
                            vehicles, self.camera_FR
                        )
                        self.data_buffer[timestamp.frame_count].transform = self.car.get_transform()
                        self.data_buffer[timestamp.frame_count].velocity = self.car.get_velocity()
                        self.data_buffer[timestamp.frame_count].angular_velocity = self.car.get_angular_velocity()
                        self.data_buffer[timestamp.frame_count].acceleration = self.car.get_acceleration()
                        self.data_buffer[timestamp.frame_count].is_at_traffic_light = self.car.is_at_traffic_light()

                        self.aggregate_buffer()
                    else:
                        print("skipping")

                    if self.visualize:
                        pygame.display.flip()

            callback_id = self.world.on_tick(on_tick)

            t = time.time()
            for i in range(10 * self.session_length):  # 3 minutes
                self.world.tick(60)
                if self.visualize:
                    pygame.event.pump()
                print("current fps is ", i / (time.time() - t))

            return True

        except Exception as e:
            print(e)
            traceback.print_exc()

            return False

        finally:
            self.world.remove_on_tick(callback_id)
            with mutex:  # acquire this to prevent ticks from happening
                print("terminating, destroying sensors and agens")

                self.camera_FC.stop()
                self.camera_FC.destroy()

                self.camera_FR.stop()
                self.camera_FR.destroy()

                self.camera_FL.stop()
                self.camera_FL.destroy()

                self.lidar_FC.stop()
                self.lidar_FC.destroy()

                self.lidar_FR.stop()
                self.lidar_FR.destroy()

                self.lidar_FL.stop()
                self.lidar_FL.destroy()

                self.car.destroy()
                self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors])

                if self.visualize:
                    pygame.quit()

    def aggregate_buffer(self):
        items = list(self.data_buffer.items())
        print("buffer length is", len(items))
        for k, v in items:
            if v.is_complete():
                self.process_data(k, v)
                del self.data_buffer[k]

    def process_sensor_pair(self, camera, boxes, camera_image, lidar_data):
        """Given data, builds a dict with images and boxes to be saved"""

        image_raw = camera_image
        image = np.frombuffer(image_raw.raw_data, dtype=np.dtype("uint8"))
        image = np.reshape(image, (image_raw.height, image_raw.width, 4))
        image_raw.convert(COLOR_CONVERTERS[CAMERA_BP_STR])

        bboxes_q = ClientSideBoundingBoxes.check_bboxes_consistency_lidar(boxes, camera, lidar_data, 200, 5)
        clean_boxes = []
        for b, q in zip(boxes, bboxes_q):
            if q:
                res = dict(b._asdict())
                del res["vehicle"]
                clean_boxes.append(res)

        return (image, clean_boxes)

    def process_data(self, key: str, values: WorldBoxInfo):
        path = self.current_run_dir / f"{key}_labels.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        image_fc, boxes_fc = self.process_sensor_pair(
            self.camera_FC, values.boxes_FC, values.camera_image_FC, values.lidar_data_FC
        )
        image_fl, boxes_fl = self.process_sensor_pair(
            self.camera_FL, values.boxes_FL, values.camera_image_FL, values.lidar_data_FL
        )
        image_fr, boxes_fr = self.process_sensor_pair(
            self.camera_FR, values.boxes_FR, values.camera_image_FR, values.lidar_data_FR
        )

        result = dict(
            transform=dict(
                x=values.transform.location.x,
                y=values.transform.location.y,
                z=values.transform.location.z,
                roll=values.transform.rotation.roll,
                pitch=values.transform.rotation.pitch,
                yaw=values.transform.rotation.yaw,
            ),
            velocity=dict(
                x=values.velocity.x,
                y=values.velocity.y,
                z=values.velocity.z,
            ),
            angular_velocity=dict(
                x=values.angular_velocity.x,
                y=values.angular_velocity.y,
                z=values.angular_velocity.z,
            ),
            acceleration=dict(
                x=values.acceleration.x,
                y=values.acceleration.y,
                z=values.acceleration.z,
            ),
            is_at_traffic_light=values.is_at_traffic_light,
            boxes=dict(
                FC=boxes_fc,
                FR=boxes_fr,
                FL=boxes_fl,
            ),
        )
        imageio.imsave(path.with_name(f"{key}_FC.png"), image_fc)
        imageio.imsave(path.with_name(f"{key}_FL.png"), image_fl)
        imageio.imsave(path.with_name(f"{key}_FR.png"), image_fr)

        with open(path, "w") as f:
            json.dump(
                result,
                f,
            )

        self.render(self.display, image_fr, boxes_fr)


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Generate carla data")

    parser.add_argument(
        "-d",
        "--dest",
        type=str,
        help="Destination folder to put the data in.",
        default="./output",
    )

    parser.add_argument(
        "-m",
        "--map",
        type=int,
        help="Map number to be loaded.",
        default=4,
    )

    parser.add_argument(
        "-c",
        "--traffic-density",
        type=float,
        help="Density of traffic - fraction of spawn points occupied.",
        default=0.5,
    )

    parser.add_argument(
        "-s",
        "--seconds",
        type=int,
        help="Number of seconds in each sessions.",
        default=180,
    )

    parser.add_argument(
        "-n",
        "--num-sessions",
        type=int,
        help="Number of sessions.",
        default=100,
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="If set, creates a window to show the car.",
    )

    args = parser.parse_args()
    return args


def main():
    """Run main loop."""
    args = parse_args()
    print(args)
    for _ in range(args.num_sessions):
        client = BasicSynchronousClient(
            map_id=args.map,
            traffic_density=args.traffic_density,
            session_length=args.seconds,
            destination_dir=args.dest,
            visualize=args.visualize,
        )
        if not client.game_loop():
            break


if __name__ == "__main__":
    main()
