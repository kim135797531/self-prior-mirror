import copy
from collections import deque
from typing import Dict
import numpy as np
from gymnasium import spaces
from einops import rearrange
from program.envs.isi_env import ISIEnv
from dataclasses import dataclass


@dataclass
class QueueData:
    joint_data_qpos: np.ndarray
    sticker_pos: np.ndarray
    sticker_rgba: np.ndarray


class DelayedQueue:
    def __init__(self, maxlen):
        self.queue = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def reset(self):
        self.queue.clear()

    def push(self, data):
        self.queue.append(data)

    def get(self):
        return None if len(self.queue) == 0 else self.queue[0]


class ArmMirrorEnv(ISIEnv):

    def __init__(
        self,
        initial_joint_qpos=None,
        detach_step_threshold=1,
        mirror_latency=0,
        sticker_detach_distance_threshold=None,
        **kwargs,
    ):
        if "model_path" not in kwargs:
            kwargs["model_path"] = "./env_arm_mirror.xml"
        if "render_mode" not in kwargs:
            kwargs["render_mode"] = "rgb_array"

        if "width" in kwargs:
            vision_width = kwargs["width"]
        else:
            vision_width = 64

        kwargs["width"] = vision_width
        kwargs["height"] = vision_width

        ISIEnv.__init__(
            self,
            initial_joint_qpos=initial_joint_qpos,
            **kwargs,
        )
        self.mujoco_renderer.render_flags = dict(
            flags=dict(
                VIS_ACTUATOR=0, VIS_TRANSPARENT=0, VIS_CONTACTFORCE=0, RND_SKYBOX=1
            ),
            vopt=dict(
                geomgroup=[1, 1, 1, 1, 1, 0],
                sitegroup=[1, 1, 1, 1, 1, 0],
                joint_group=[1, 1, 1, 1, 1, 0],
                tendongroup=[1, 1, 1, 1, 1, 0],
                actuatorgroup=[1, 1, 1, 1, 1, 0],
                skingroup=[1, 1, 1, 1, 1, 1],
            ),
        )

        """
        Define hidden states
        """
        self._has_sticker = None
        self.baby_sticker = self.model.geom("baby_sticker")
        self.baby_sticker_data = self.data.geom("baby_sticker")  # good pos = 0 0.1 0
        self.baby_latency_queue = DelayedQueue(mirror_latency)
        self.mirror_sticker = self.model.geom("mirror_sticker")
        self.mirror_sticker_data = self.data.geom("mirror_sticker")
        self.mirror_latency = mirror_latency

        """
        Define environment shape
        """
        self.baby_touch_sensor = self.model.sensor("baby_touch_sensor")
        self.baby_touch_sensor_data = self.data.sensor("baby_touch_sensor")

        # Nearest distance from hand to sticker = 0.02 = 2cm
        if sticker_detach_distance_threshold < 0:
            sticker_detach_distance_threshold = 0.02

        self.sticker_detach_distance_threshold = sticker_detach_distance_threshold
        print(
            "Sticker detach distance threshold:", self.sticker_detach_distance_threshold
        )

        # 10 steps (0.1s) = success (horizon 15 -> 1 process = below 24GB)
        # 50 steps (0.5s) = success (horizon 50 -> 1 process = 46GB)
        # 100 steps (1s) = failed..
        self.sticker_detach_step_threshold = detach_step_threshold

        """
        Define observations
        """
        # Vision
        vision_height = vision_width
        self.baby_r_camera = self.model.camera("baby_r_camera")
        self.vision = np.zeros((3, vision_height, vision_width), dtype=np.uint8)
        self.world_camera = self.model.camera("world_camera")
        self.world_vision = np.zeros((3, vision_height, vision_width), dtype=np.uint8)

        # Proprioception
        self.baby_r_upper_arm_lift = self.model.joint("baby_r_upper_arm_lift")
        self.baby_r_upper_arm_roll = self.model.joint("baby_r_upper_arm_roll")
        self.baby_r_upper_arm_flex = self.model.joint("baby_r_upper_arm_flex")
        self.baby_r_lower_arm_flex = self.model.joint("baby_r_lower_arm_flex")
        self.baby_r_hand_flex = self.model.joint("baby_r_hand_flex")
        self.baby_joints = [
            self.baby_r_upper_arm_lift,
            self.baby_r_upper_arm_roll,
            self.baby_r_upper_arm_flex,
            self.baby_r_lower_arm_flex,
            self.baby_r_hand_flex,
        ]
        self.baby_joint_datas = [
            self.data.joint("baby_r_upper_arm_lift"),
            self.data.joint("baby_r_upper_arm_roll"),
            self.data.joint("baby_r_upper_arm_flex"),
            self.data.joint("baby_r_lower_arm_flex"),
            self.data.joint("baby_r_hand_flex"),
        ]
        self.mirror_joint_datas = [
            self.data.joint("mirror_l_upper_arm_lift"),
            self.data.joint("mirror_l_upper_arm_roll"),
            self.data.joint("mirror_l_upper_arm_flex"),
            self.data.joint("mirror_l_lower_arm_flex"),
            self.data.joint("mirror_l_hand_flex"),
        ]
        n_baby_joints = len(self.baby_joints)
        self.proprioception = np.zeros(n_baby_joints, dtype=np.float32)

        """
        Define observation specs for Gymnasium
        """
        self.min_proprioception = np.zeros(n_baby_joints, dtype=np.float32)
        self.max_proprioception = np.zeros(n_baby_joints, dtype=np.float32)
        for i, baby_joint in enumerate(self.baby_joints):
            self.min_proprioception[i] = baby_joint.range[0]
            self.max_proprioception[i] = baby_joint.range[1]

        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(
                    low=0,
                    high=255,
                    shape=self.vision.shape,
                    dtype=np.uint8,
                ),
                "proprioception": spaces.Box(
                    low=self.min_proprioception,
                    high=self.max_proprioception,
                    shape=(n_baby_joints,),
                    dtype=np.float32,
                ),
            }
        )

        """
        Define actions
        """
        self.min_action = -2.0  # Defined in xml
        self.max_action = 2.0
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(n_baby_joints,),
            dtype=np.float32,
        )

        """
        Cached values
        """
        self.last_action = None
        self.is_sticker_detached = False
        self.sticker_detach_step_count = None
        self.skip_render = False

        """
        Dummy data
        """
        self._reset_data()

    def _reset_data(self):
        self._has_sticker = False
        self.last_action = None
        self.is_sticker_detached = False
        self.sticker_detach_step_count = 0
        self.baby_latency_queue.reset()

    def set_random_pose(self, init_random_steps):
        # Randomize joint position by give random action
        action = self.action_space.sample()

        for t in range(init_random_steps):
            _ = ISIEnv.step(self, action)

    def set_random_sticker_xpos(self):
        self.baby_sticker.pos[0] = self.np_random.uniform(low=0.0, high=0.05, size=1)[0]
        self.baby_sticker.pos[1] = 0.12
        self.baby_sticker.pos[2] = self.np_random.uniform(low=0.0, high=0.15, size=1)[0]

    def _update_mirror(self):
        # Read delayed data first.
        delayed_data = self.baby_latency_queue.get()

        # Then push current state into the queue.
        joint_data_qpos = np.zeros_like(self.proprioception)
        sticker_pos = np.zeros(3, dtype=np.float32)
        sticker_rgba = np.zeros(4, dtype=np.float32)

        for i in range(len(self.baby_joints)):
            joint_data_qpos[i] = self.baby_joint_datas[i].qpos[0]

        sticker_pos[:] = self.baby_sticker.pos[:]
        sticker_rgba[:] = self.baby_sticker.rgba[:]

        baby_data = QueueData(joint_data_qpos, sticker_pos, sticker_rgba)
        self.baby_latency_queue.push(baby_data)

        # Apply delayed state to mirror joints/sticker.
        if delayed_data is None:
            delayed_data = baby_data

        for i in range(len(self.baby_joints)):
            self.mirror_joint_datas[i].qpos[:] = delayed_data.joint_data_qpos[i]

        self.mirror_sticker.pos[0] = -1 * delayed_data.sticker_pos[0]
        self.mirror_sticker.pos[1] = delayed_data.sticker_pos[1]
        self.mirror_sticker.pos[2] = delayed_data.sticker_pos[2]
        self.mirror_sticker.rgba[:] = delayed_data.sticker_rgba[:]

    def _update_proprioception(self):
        for i, baby_joint_data in enumerate(self.baby_joint_datas):
            self.proprioception[i] = baby_joint_data.qpos[0]

        self.proprioception = np.clip(
            self.proprioception,
            self.min_proprioception,
            self.max_proprioception,
        )

    def _update_sticker(self):
        if self._has_sticker:
            self.baby_sticker.rgba[3] = 1.0
        else:
            self.baby_sticker.rgba[3] = 0.0  # Invisible
            if self.is_sticker_detached:
                self.baby_sticker.pos[0] = 99  # Out of sight

    def _distance_hand_to_sticker(self):
        distance = self.baby_touch_sensor_data.data[0]
        return distance if distance >= 0 else 0  # Ignore negative distance

    def _update_should_sticker_detach(self, ignore_counters):
        distance = self._distance_hand_to_sticker()
        near_enough = distance < self.sticker_detach_distance_threshold

        if not near_enough:
            self.sticker_detach_step_count = 0
            return

        if not ignore_counters:
            self.sticker_detach_step_count += 1

        if self.sticker_detach_step_threshold < self.sticker_detach_step_count:
            self._has_sticker = False
            self.is_sticker_detached = True

    def _update_vision(self):
        if self.skip_render:
            return

        if self.render_mode == "rgb_array":
            self.mujoco_renderer.camera_id = self.baby_r_camera.id
            self.vision[:] = rearrange(self.render(), "h w c -> c h w")  # height first!

            self.mujoco_renderer.camera_id = self.world_camera.id
            self.world_vision[:] = rearrange(self.render(), "h w c -> c h w")
        else:
            # For human mode, just print black array
            pass

    def _get_obs(self):
        return {
            "vision": self.vision.copy(),
            "proprioception": self.proprioception.copy(),
        }

    def _get_info(self):
        return {
            "has_sticker": self._has_sticker,
            "is_sticker_detached": self.is_sticker_detached,
            "world_vision": self.world_vision.copy(),
        }

    def _get_reset_info(self) -> Dict[str, float]:
        super_info = ISIEnv._get_reset_info(self)
        info = self._get_info()
        return super_info | info

    def reset_model(self):
        # d-kim: We need to split as reset_model, because we don't have np_random until reset() in gym Env
        # super_observation, super_info = ISIEnv.reset_model(self)
        ISIEnv.reset_model(self)

    def reset(self, seed=None, options=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        super_observation, super_info = ISIEnv.reset(self, seed=seed, options=options)
        self._reset_data()

        # Randomize joint position by give random action
        init_random_steps = options.get("init_random_steps")
        self.set_random_pose(init_random_steps)

        self.set_random_sticker_xpos()

        # Reset cached values
        self.last_action = self.action_space.sample()

        options = options if options else dict()
        has_sticker = options.get("has_sticker")
        sticker_xpos = options.get("sticker_xpos")
        imitation_queue = options.get("imitation_queue")

        if has_sticker is not None:
            self._has_sticker = has_sticker

        if sticker_xpos is not None:
            raise NotImplementedError("sticker_xpos is not implemented yet")

        if imitation_queue is not None:
            self.baby_latency_queue = imitation_queue
            self.mirror_latency = imitation_queue.maxlen

        self.forward()

        observation = self._get_obs()
        info = self._get_reset_info()

        return observation, info

    def step(self, action):
        self.last_action = action

        (
            super_observation,
            super_reward,
            super_terminated,
            super_truncated,
            super_info,
        ) = ISIEnv.step(self, action)
        self.forward()

        distance = self._distance_hand_to_sticker()
        near_enough = distance < self.sticker_detach_distance_threshold

        assert not (self._has_sticker and self.is_sticker_detached)

        if self._has_sticker and not near_enough:
            reward = 0
        elif self._has_sticker and near_enough:
            reward = (
                self.sticker_detach_distance_threshold - distance
            ) / self.sticker_detach_distance_threshold
        elif not self._has_sticker and self.is_sticker_detached:
            reward = 1
        else:
            reward = 0

        observation = self._get_obs()
        info = self._get_info()
        info = super_info | info

        reward = super_reward + reward
        terminated = super_terminated
        truncated = super_truncated

        return observation, reward, terminated, truncated, info

    def forward(self, ignore_counters=False):
        # Update hand and sticker positions.
        ISIEnv.forward(self)

        # Update proprioception from joint qpos.
        self._update_proprioception()

        # Compute distance with the updated positions.
        if self._has_sticker:
            self._update_should_sticker_detach(ignore_counters=ignore_counters)

        # Refresh sticker RGBA state.
        self._update_sticker()

        if not ignore_counters:
            # Update mirror state as well.
            self._update_mirror()

        # Apply sticker color update in simulator state.
        ISIEnv.forward(self)

        # RGB render
        self._update_vision()


if __name__ == "__main__":
    render_mode = "human"
    width = 640
    # render_mode = "rgb_array"
    # width = 64
    env = ArmMirrorEnv(
        render_mode=render_mode,
        width=width,
        detach_step_threshold=50,
        mirror_latency=0,
    )
    env.reset(options={"has_sticker": True, "init_random_steps": 100})

    if render_mode == "rgb_array":
        import matplotlib.pyplot as plt

    for i in range(10000000000000):
        if render_mode == "rgb_array":
            if i % 100 == 0:
                img_cam = env.render()
                plt.figure()
                plt.imshow(img_cam)

                # env.unwrapped.mujoco_renderer.camera_id = -1
                # img_track = env.render()
                # plt.figure()
                # plt.imshow(img_track)

                plt.show()
                print(i)
        else:
            if i % 1000 == 0:
                env._has_sticker = True
                env.is_sticker_detached = False
                env.set_random_sticker_xpos()
                env.set_random_pose(100)
            env.render()

        env.step(np.zeros(env.action_space.shape))
