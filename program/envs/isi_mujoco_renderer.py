#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""isi_mujoco_renderer.py
Created by Dongmin Kim at 2023-10-17

Modified from
https://github.com/rohanpsingh/mujoco-python-viewer
"""
import numba  # Numba should be imported before mujoco
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, OffScreenViewer
from program.envs.isi_window_viewer import ISIWindowViewer
from typing import Optional


class ISIMujocoRenderer(MujocoRenderer):
    """d-kim: Almost same with original MujocoRenderer in gymnasium,
    but appling render flags to viewer, and using custom GUI renderer"""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        render_flags: dict,
        default_cam_config: Optional[dict] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_geom: int = 50000,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        super().__init__(
            model,
            data,
            default_cam_config,
            width,
            height,
            max_geom,
            camera_id,
            camera_name,
        )
        self.render_flags = render_flags

        if camera_id is not None:
            self.camera_id = camera_id

    def apply_render_flags(self, viewer):
        vopt = viewer.vopt
        vopt.geomgroup[0:6] = self.render_flags["vopt"]["geomgroup"]
        vopt.sitegroup[0:6] = self.render_flags["vopt"]["sitegroup"]
        vopt.jointgroup[0:6] = self.render_flags["vopt"]["joint_group"]
        vopt.tendongroup[0:6] = self.render_flags["vopt"]["tendongroup"]
        vopt.actuatorgroup[0:6] = self.render_flags["vopt"]["actuatorgroup"]
        vopt.skingroup[0:6] = self.render_flags["vopt"]["skingroup"]

        flags = vopt.flags
        flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = self.render_flags["flags"][
            "VIS_ACTUATOR"
        ]
        flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = self.render_flags["flags"][
            "VIS_TRANSPARENT"
        ]
        flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.render_flags["flags"][
            "VIS_CONTACTFORCE"
        ]

        scn = viewer.scn
        scn_flags = scn.flags
        scn_flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = self.render_flags["flags"][
            "RND_SKYBOX"
        ]

    def _get_viewer(self, render_mode: str):
        """Override upper method to use ISI Custom renderer"""
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = ISIWindowViewer(
                    self.model, self.data, self.width, self.height, self.max_geom
                )

            elif render_mode in {"rgb_array", "depth_array"}:
                self.viewer = OffScreenViewer(
                    self.model, self.data, self.width, self.height, self.max_geom
                )
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, or depth_array"
                )
            # Add default camera parameters
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

        self.apply_render_flags(self.viewer)
        return self.viewer
