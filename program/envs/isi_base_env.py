import os
from abc import ABC
from typing import Any, Dict, Optional, Tuple, Union
import numba  # Numba should be imported before mujoco
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space, Box
from program.envs.isi_mujoco_renderer import ISIMujocoRenderer


class ISIBaseEnv(MujocoEnv, ABC):
    # Abstract class
    GEOM_GROUP_DEFAULT = 0
    GEOM_GROUP_BONE = 1
    GEOM_GROUP_RIGID_SKIN = 2
    GEOM_GROUP_SOFT_SKIN = 3
    GEOM_GROUP_ENV = 4
    GEOM_GROUP_WRAP_OBJECT = 5

    SITE_GROUP_DEFAULT = 0
    SITE_GROUP_TACTILE_MIDPOINT = 1
    SITE_GROUP_TACTILE_RANGE = 2
    SITE_GROUP_MUSCLE_POINT = 3
    SITE_GROUP_SOFT_SKIN = 4
    SITE_GROUP_ENV = 5

    TENDON_GROUP_DEFAULT = 0
    TENDON_GROUP_LARM = 1
    TENDON_GROUP_RARM = 2
    TENDON_GROUP_LLEG = 3
    TENDON_GROUP_RLEG = 4
    TENDON_GROUP_ENV = 5

    ACTUATOR_GROUP_DEFAULT = 0
    ACTUATOR_GROUP_LARM = 1
    ACTUATOR_GROUP_RARM = 2
    ACTUATOR_GROUP_LLEG = 3
    ACTUATOR_GROUP_RLEG = 4
    ACTUATOR_GROUP_ENV = 5

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        initial_joint_qpos=None,
        default_camera_config: Optional[dict] = None,
        **kwargs,
    ):
        self.initial_joint_qpos = (
            dict() if initial_joint_qpos is None else initial_joint_qpos
        )

        self.render_flags = dict(
            flags=dict(
                VIS_ACTUATOR=1, VIS_TRANSPARENT=1, VIS_CONTACTFORCE=0, RND_SKYBOX=1
            ),
            vopt=dict(
                geomgroup=[1, 1, 1, 0, 1, 0],
                sitegroup=[1, 1, 0, 0, 0, 0],
                joint_group=[1, 1, 1, 1, 1, 0],
                tendongroup=[1, 1, 1, 1, 1, 0],
                actuatorgroup=[1, 1, 1, 1, 1, 0],
                skingroup=[1, 1, 1, 0, 0, 0],
            ),
        )

        # 231017: We don't use observation space concept,
        #  Instead, user should decide what to use in info dict.
        self.dummy_obs = np.zeros(1, dtype=np.float64)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

        error_in_model_path = True
        isi_asset_path = os.path.dirname(os.path.abspath(__file__))
        isi_asset_path = f"{isi_asset_path}/../assets"
        if "model_path" in kwargs:
            model_path = kwargs["model_path"]
            if model_path.startswith("isi://"):
                model_path = model_path.replace("isi://", "")
                model_path = f"{isi_asset_path}/{model_path}"
                kwargs["model_path"] = model_path
                error_in_model_path = False
            elif (
                model_path.startswith(".")
                or model_path.startswith("/")
                or model_path.startswith("~")
            ):
                error_in_model_path = False

        if error_in_model_path:
            xmls = os.listdir(isi_asset_path)
            xmls = sorted([f"isi://{xml}" for xml in xmls if xml.endswith(".xml")])
            print(
                "model_path should be given, and should be starts with 'isi://'. \n"
                "Available models: "
            )
            print(*xmls, sep="\n")
            raise Exception()

        MujocoEnv.__init__(
            self, frame_skip=1, observation_space=observation_space, **kwargs
        )
        self.metadata["render_fps"] = int(1 / self.dt)
        self.mujoco_renderer = ISIMujocoRenderer(
            self.model,
            self.data,
            self.render_flags,
            default_camera_config,
            self.width,
            self.height,
            max_geom=50000,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
        )

        # Check if this env has ISICallback feature, and then disable warmstart if true.
        if "callback" in str(self.__class__):
            if self.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_WARMSTART == 0:
                errmsg = (
                    "d-kim: ISICallback can't be used with warmstart, \n"
                    "because we call isi_forward a lot. \n"
                    "(mj_forward changes simulation result if warmstart is enabled) \n"
                    "You should add a flag in xml as \n"
                    "<option><flag warmstart='disable'></option> \n"
                    "Warmstart was disabled temporally for this simulation."
                )
                print(errmsg)
                self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART

        self.time_step = self.model.opt.timestep

        def _ids2names(mj_type, length):
            ret = []
            for i in range(length):
                ret.append(mujoco.mj_id2name(self.model, mj_type, i))
            return ret

        self.body_names = _ids2names(mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)
        self.joint_names = _ids2names(mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)
        self.geom_names = _ids2names(mujoco.mjtObj.mjOBJ_GEOM, self.model.ngeom)
        self.site_names = _ids2names(mujoco.mjtObj.mjOBJ_SITE, self.model.nsite)
        self.camera_names = _ids2names(mujoco.mjtObj.mjOBJ_CAMERA, self.model.ncam)
        self.light_names = _ids2names(mujoco.mjtObj.mjOBJ_LIGHT, self.model.nlight)
        self.mesh_names = _ids2names(mujoco.mjtObj.mjOBJ_MESH, self.model.nmesh)
        self.equality_names = _ids2names(mujoco.mjtObj.mjOBJ_EQUALITY, self.model.neq)
        self.tendon_names = _ids2names(mujoco.mjtObj.mjOBJ_TENDON, self.model.ntendon)
        self.actuator_names = _ids2names(mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu)
        self.sensor_names = _ids2names(mujoco.mjtObj.mjOBJ_SENSOR, self.model.nsensor)
        self.numeric_names = _ids2names(
            mujoco.mjtObj.mjOBJ_NUMERIC, self.model.nnumeric
        )
        self.text_names = _ids2names(mujoco.mjtObj.mjOBJ_TEXT, self.model.ntext)
        self._np_site_jacp = np.zeros((self.model.nsite, 3, self.model.nv))

    @property
    def site_jacp(self):
        for i, jacp in enumerate(self._np_site_jacp):
            mujoco.mj_jacSite(self.model, self.data, jacp, None, i)
        return self._np_site_jacp

    @property
    def site_xvelp(self):
        jacp = self.site_jacp.reshape((self.model.nsite, 3, self.model.nv))
        xvelp = np.dot(jacp, self.data.qvel)
        return xvelp

    def get_n_tactile_sensors(self):
        # TODO: 211006: d-kim: Actually, we are using (and calculating) the custom
        # tactile sensor from the "mujoco site (ex: only_display_Rhead_0)",
        # but currently the number of that site and "mujoco touch sensor" is same,
        # so we are using just number of "mujoco touch sensor"
        return np.sum(self.model.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)

    def get_visible_joint_names(self):
        return self.model.joint_names

    def get_visible_joint_original_joint_names(self):
        return self.get_visible_joint_names()

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        # super_obs = super().step(action)  # Not neccessary if this is top-level class

        self.do_simulation(action, self.frame_skip)

        observation = self.__get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = self.__get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def __get_obs(self):
        return self.dummy_obs.copy()

    def __get_info(self):
        info = dict()
        info["mj_joint_pos"] = self.data.qpos.copy()
        info["mj_joint_vel"] = self.data.qvel.copy()
        info["name_mj_joint"] = self.joint_names
        info["name_mj_joint_pos"] = self.joint_names
        info["name_mj_joint_vel"] = self.joint_names
        info["name_mj_joint_acc"] = self.joint_names
        return info

    def _get_reset_info(self) -> Dict[str, float]:
        return self.__get_info()

    def reset_model(self):
        # super_obs = super().reset_model()  # Not neccessary if this is top-level class
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        for joint_id, joint_qpos in self.initial_joint_qpos.items():
            qpos[joint_id] = joint_qpos

        self.set_state(qpos, qvel)
        self.forward()

        observation = self.__get_obs()
        return observation
