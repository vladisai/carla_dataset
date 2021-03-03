#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
from carla import ColorConverter as cc
import logging
import threading
import pickle
import string
import random
from datetime import datetime
from pathlib import Path

mutex = threading.Lock()


def breakpoint():
    import pdb

    pdb.set_trace()


try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (sys.version_info.major, sys.version_info.minor, "win-amd64" if os.name == "nt" else "linux-x86_64")
        )[0]
    )
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

CAMERA_ATTRIBUTES = {
    "image_size_x": str(VIEW_WIDTH),
    "image_size_y": str(VIEW_HEIGHT),
    "fov": str(VIEW_FOV),
}

BB_COLOR = (248, 64, 24)
BB_COLOR_Q = (64, 248, 24)

CAMERA_BP_STR = "sensor.camera.rgb"
SEMANTIC_BP_STR = "sensor.camera.semantic_segmentation"
DEPTH_BP_STR = "sensor.camera.depth"

COLOR_CONVERTERS = {
    CAMERA_BP_STR: cc.Raw,
    SEMANTIC_BP_STR: cc.CityScapesPalette,
    DEPTH_BP_STR: cc.Depth,
}

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


def spawn_vehicles(client, world, number_of_vehicles):
    blueprints = world.get_blueprint_library().filter("vehicle.*")

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        msg = "requested %d vehicles, but could only find %d spawn points"
        logging.warning(msg, number_of_vehicles, number_of_spawn_points)
        number_of_vehicles = number_of_spawn_points

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
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

    for response in client.apply_batch_sync(batch):
        if response.error:
            logging.error(response.error)
        else:
            actor_list.append(response.actor_id)

    return actor_list, spawn_points[n:]


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

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
    def draw_bounding_boxes(bb_surface, bounding_boxes, boxes_q):
        """
        Draws bounding boxes on pygame display.
        """

        # bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox, q in zip(bounding_boxes, boxes_q):
            points = bbox.points
            # draw lines
            # base
            if q:
                color = BB_COLOR
            else:
                color = BB_COLOR_Q

            pygame.draw.line(bb_surface, color, points[0], points[1])
            pygame.draw.line(bb_surface, color, points[0], points[1])
            pygame.draw.line(bb_surface, color, points[1], points[2])
            pygame.draw.line(bb_surface, color, points[2], points[3])
            pygame.draw.line(bb_surface, color, points[3], points[0])

        # display.blit(bb_surface, (0, 0))

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

            res = Box(points, distance_to_camera, max_distance_to_center)

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
        """
        Creates matrix from carla transform.
        """

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


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================

from collections import defaultdict, namedtuple

Box = namedtuple("Box", ["points", "distance_to_camera", "max_distance_to_center"])


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True

        self.camera_attributes = {
            "image_size_x": str(VIEW_WIDTH),
            "image_size_y": str(VIEW_HEIGHT),
            "fov": str(VIEW_FOV),
        }

        self.data_buffer = dict()

    def blueprint(self, bp_str):
        camera_bp = self.world.get_blueprint_library().find(bp_str)
        for k, v in CAMERA_ATTRIBUTES.items():
            camera_bp.set_attribute(k, v)
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self, spawns):
        """
        Spawns actor-vehicle to be controled.
        """
        # self.car = self.world.get_actors([car_id])[0]

        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")[0]
        car_bp.set_attribute("role_name", "autopilot")
        location = random.choice(spawns)
        self.car = self.world.spawn_actor(car_bp, location)
        self.car.set_autopilot(True)

    def setup_sensor(self, bp_str, visualize=False):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        # camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=0.9, z=1.4), carla.Rotation(pitch=0))
        bp = self.blueprint(bp_str)
        # bp.set_attribute("sensor_tick", "0.1")
        sensor = self.world.spawn_actor(bp, camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)

        data_lambda = lambda data: weak_self().new_data(weak_self, bp_str, data)

        if visualize:
            visusalization_lambda = lambda image: weak_self().set_image(weak_self, image, COLOR_CONVERTERS[bp_str])
            joint_lambda = lambda image: (visusalization_lambda(image), data_lambda(image))
            sensor.listen(joint_lambda)
        else:
            sensor.listen(data_lambda)

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        sensor.calibration = calibration
        return sensor

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
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
    def set_image(weak_self, img, cc):
        # points = np.frombuffer(img.raw_data, dtype=np.dtype('d4')).reshape((VIEW_HEIGHT, VIEW_WIDTH))
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            img.convert(cc)
            self.image = img
            self.capture = False

    @staticmethod
    def new_data(weak_self, bp_str, data):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if data.frame_number not in self.data_buffer:
            self.data_buffer[data.frame_number] = dict()

        self.data_buffer[data.frame_number][bp_str] = data

    def render(self, display, image, boxes, boxes_q):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.image.height, self.image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        ClientSideBoundingBoxes.draw_bounding_boxes(surface, boxes, boxes_q)

        display.blit(surface, (0, 0))

    def game_loop(self):
        """
        Main program loop.

        """

        date_time = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

        self.run_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + date_time
        print('run id is', self.run_id)

        try:
            pygame.init()

            self.client = carla.Client("127.0.0.1", 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            actors, spawns = spawn_vehicles(self.client, self.world, 50)
            self.setup_car(spawns)

            self.camera = self.setup_sensor(CAMERA_BP_STR)
            self.semantic = self.setup_sensor(SEMANTIC_BP_STR)
            self.depth = self.setup_sensor(DEPTH_BP_STR, visualize=True)

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter("vehicle.*")

            def on_tick(timestamp):
                with mutex:
                    self.capture = True

                    bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                    if timestamp.frame_count not in self.data_buffer:
                        self.data_buffer[timestamp.frame_count] = dict()

                    self.data_buffer[timestamp.frame_count]["bbox"] = bounding_boxes

                    self.data_buffer[timestamp.frame_count]["transform"] = self.car.get_transform()
                    self.data_buffer[timestamp.frame_count]["velocity"] = self.car.get_velocity()
                    self.data_buffer[timestamp.frame_count]["angular_velocity"] = self.car.get_angular_velocity()
                    self.data_buffer[timestamp.frame_count]["acceleration"] = self.car.get_acceleration()
                    print(
                        "acceleration",
                        self.car.get_transform(),
                        self.car.get_velocity(),
                        self.car.get_acceleration(),
                        self.car.get_angular_velocity(),
                    )

                    pygame.display.flip()

                    pygame.event.pump()
                    if self.control(self.car):
                        return

                    self.aggregate_buffer()

            self.world.on_tick(on_tick)

            while True:
                self.world.tick()
                pygame_clock.tick_busy_loop(1)

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.semantic.destroy()
            self.depth.destroy()
            self.car.destroy()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in actors])
            pygame.quit()

    def aggregate_buffer(self):
        items = list(self.data_buffer.items())
        cnt = 0
        for k, v in items:
            if len(v) == 8:
                self.process_data(k, v)
                del self.data_buffer[k]
                # import pdb; pdb.set_trace()
            else:
                cnt += 1

    def process_data(self, key, values):
        bboxes = values["bbox"]

        image_raw = values[CAMERA_BP_STR]
        image = np.frombuffer(image_raw.raw_data, dtype=np.dtype("uint8"))
        image = np.reshape(image, (image_raw.height, image_raw.width, 4))

        semantic_raw = values[SEMANTIC_BP_STR]
        semantic = np.frombuffer(semantic_raw.raw_data, dtype=np.dtype("uint8")).copy()
        semantic = np.reshape(semantic, (semantic_raw.height, semantic_raw.width, 4))[:, :, 2]
        semantic_raw.convert(COLOR_CONVERTERS[SEMANTIC_BP_STR])

        depth_raw = values[DEPTH_BP_STR]
        depth_raw.convert(COLOR_CONVERTERS[DEPTH_BP_STR])
        depth = np.frombuffer(depth_raw.raw_data, dtype=np.dtype("uint8")).copy()
        depth = np.reshape(depth, (depth_raw.height, depth_raw.width, 4))
        depth_raw_buf = depth.copy()
        depth = np.dot(depth[:, :, :3], [65536.0, 256.0, 1.0])
        depth_normalized = depth / 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        depth = depth_normalized * 1000.0

        bboxes_q = [check_bbox_consistency(bbox, image, semantic, depth) for bbox in bboxes]

        # import pdb
        # pdb.set_trace()

        path = Path("outputs") / self.run_id / f"{key}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                dict(
                    image=image,
                    semantic=semantic,
                    depth=depth,
                    depth_raw=depth_raw_buf,
                    transform=dict(
                        x=values["transform"].location.x,
                        y=values["transform"].location.y,
                        z=values["transform"].location.z,
                        roll=values["transform"].rotation.roll,
                        pitch=values["transform"].rotation.pitch,
                        yaw=values["transform"].rotation.yaw,
                    ),
                    velocity=dict(
                        x=values["velocity"].x,
                        y=values["velocity"].y,
                        z=values["velocity"].z,
                    ),
                    angular_velocity=dict(
                        x=values["angular_velocity"].x,
                        y=values["angular_velocity"].y,
                        z=values["angular_velocity"].z,
                    ),
                    acceleration=dict(
                        x=values["acceleration"].x,
                        y=values["acceleration"].y,
                        z=values["acceleration"].z,
                    ),
                    bboxes=bboxes,
                    bboxes_q=bboxes_q,
                ),
                f,
            )

        self.render(self.display, image_raw, bboxes, bboxes_q)


def check_bbox_consistency(bbox, image, semantic, depth):
    # print('distance to camera', bbox.distance_to_camera)
    if bbox.distance_to_camera > 300:
        # print('distance')
        return False

    max_x, max_y = bbox.points[0]
    min_x, min_y = bbox.points[2]
    max_x = np.clip(int(max_x), 0, VIEW_WIDTH)
    max_y = np.clip(int(max_y), 0, VIEW_HEIGHT)
    min_x = np.clip(int(min_x), 0, VIEW_WIDTH)
    min_y = np.clip(int(min_y), 0, VIEW_HEIGHT)

    mask = np.zeros_like(depth)
    mask[min_y:max_y, min_x:max_x] = 1
    total_mask = mask.sum()
    # print('total mask', total_mask)
    if total_mask == 0:
        return False

    car_mask = semantic == 10
    total_car = (car_mask * mask).sum()
    # print('total_car', total_car)
    if total_car == 0:
        # print("no car in box")
        return False
    total_car_part = total_car / total_mask

    correct_distance = np.absolute(depth - bbox.distance_to_camera) < 4 + bbox.max_distance_to_center

    consistent_depth = mask * correct_distance * car_mask
    total_consistent_depth = consistent_depth.sum()
    total_consistent_depth_part = total_consistent_depth / total_car
    print("total consistent depth part", total_consistent_depth_part)

    res = total_consistent_depth_part > 0.5 and total_car_part > 0.1
    # print(total_car_part)
    # res = total_car_part > 0.1

    # TODO: figure out why the boxes don't get detected. Maybe the box is drawn incorrectly.

    # if res:
    #     import pdb; pdb.set_trace()

    return res
    # 1. at least [thresh_a=0.5] of pixels with type car should be with correct distance.
    # 2. the pixels with correct distance should occupy at least [thresh_b=0.1] of the bbox.


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print("EXIT")


if __name__ == "__main__":
    main()
