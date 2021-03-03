# Carla Dataset Retrieval

import logging
import random

import glob
import os
import sys


try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (sys.version_info.major, sys.version_info.minor, "win-amd64" if os.name == "nt" else "linux-x86_64")
        )[0]
    )
except IndexError:
    pass

import carla

import time
import numpy as np
import cv2

count = 1


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


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


def save_data(vehicle, image, image_path_ego, image_path_topdown, pose_path):
    global count

    if not os.path.exists(image_path_ego):
        os.makedirs(image_path_ego)
    if not os.path.exists(image_path_topdown):
        os.makedirs(image_path_topdown)
    if not os.path.exists(pose_path):
        os.makedirs(pose_path)

    cv2.imwrite(os.path.join(image_path_ego, str(count) + ".jpg"), to_bgra_array(image))
    print(get_matrix(image.transform))
    np.save(os.path.join(pose_path, str(count)), get_matrix(image.transform))

    count += 1


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

    return actor_list


def main():

    actor_list = []

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)
        # birdview_producer = BirdViewProducer(
        #     client,
        #     target_size=PixelDimensions(width=500, height=500),
        #     pixels_per_meter=4,
        #     crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        # )
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        actors = spawn_vehicles(client, world, 100)
        main_actor = world.get_actors(
            [
                actors[0],
            ]
        )[0]

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("sensor_tick", "0.2")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(0, 0, 0))
        print(camera_transform)

        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=main_actor)
        camera.listen(
            lambda image: save_data(
                main_actor,
                image,
                "output/EgoView/",
                "output/TopDownView/",
                "output/Pose/",
            )
        )
        actor_list.append(camera)
        print("created %s" % camera.type_id)

        time.sleep(360)

    except Exception as e:
        print(e)

    finally:
        try:
            camera.destroy()
            client.apply_batch([carla.command.DestroyActor(x) for x in actors])
            print("done.")
        except:
            pass


if __name__ == "__main__":
    main()
