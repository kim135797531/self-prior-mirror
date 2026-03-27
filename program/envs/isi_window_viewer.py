#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""isi_glfw_gui_renderer.py
Created by Dongmin Kim at 2022-05-15

Modified from
https://github.com/rohanpsingh/mujoco-python-viewer
"""
import numba  # Numba should be imported before mujoco
import mujoco
import glfw
import sys
from threading import Lock
import numpy as np
import time
import imageio
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
from typing import Optional


class ISIWindowViewer(WindowViewer):
    """d-kim: Almost same with original WindowViewer in gymnasium,
    but enabled perturb, synced menu rendering"""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_geom: int = 50000,
    ):
        super().__init__(model, data, width, height, max_geom)

        self._left_double_click_pressed = False
        self._right_double_click_pressed = False

        self._last_left_click_time = None
        self._last_right_click_time = None

        self._gui_lock = Lock()

    def render(self):
        def pre_update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()
                sys.exit(0)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )
            return render_start

        def update():
            # update scene
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                self.pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            # marker items
            for marker in self._markers:
                self._add_marker_to_scene(marker)

            # render
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            # overlay items
            if not self._hide_menu:
                for gridpos, [t1, t2] in self._overlays.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.con,
                    )

        def post_update(render_start):
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )
            self._overlays.clear()

        if self._paused:
            while self._paused:
                render_start = pre_update()
                with self._gui_lock:
                    update()
                    glfw.swap_buffers(self.window)
                post_update(render_start)

                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                render_start = pre_update()
                with self._gui_lock:
                    update()
                    glfw.swap_buffers(self.window)
                post_update(render_start)
                self._loop_count -= 1

        # clear overlay
        self._overlays.clear()
        # clear markers
        self._markers.clear()

        # apply perturbation (should this come before mj_step?)
        self.apply_perturbations()

    def _cursor_pos_callback(
        self, window: "glfw.LP__GLFWwindow", xpos: float, ypos: float
    ):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self._button_left_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / width,
                    dy / height,
                    self.scn,
                    self.pert,
                )
            else:
                mujoco.mjv_moveCamera(
                    self.model, action, dx / width, dy / height, self.scn, self.cam
                )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window: "glfw.LP__GLFWwindow", button, act, mods):
        self._button_left_pressed = (
            button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        )
        self._button_right_pressed = (
            button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # detect a left- or right- doubleclick
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            time_diff = time_now - self._last_left_click_time
            if 0.01 < time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = time_now - self._last_right_click_time
            if 0.01 < time_diff < 0.2:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb

        # handle doubleclick
        if self._left_double_click_pressed or self._right_double_click_pressed:
            # determine selection mode
            selmode = 0
            if self._left_double_click_pressed:
                selmode = 1
            if self._right_double_click_pressed:
                selmode = 2
            if self._right_double_click_pressed and key:
                selmode = 3

            # find geom and 3D click point, get corresponding body
            width, height = self.viewport.width, self.viewport.height
            aspectratio = width / height
            relx = x / width
            rely = (self.viewport.height - y) / height
            selpnt = np.zeros((3, 1), dtype=np.float64)
            selgeom = np.zeros((1, 1), dtype=np.int32)
            selflex = np.zeros((1, 1), dtype=np.int32)
            selskin = np.zeros((1, 1), dtype=np.int32)
            selbody = mujoco.mjv_select(
                self.model,
                self.data,
                self.vopt,
                aspectratio,
                relx,
                rely,
                self.scn,
                selpnt,
                selgeom,
                selflex,
                selskin,
            )

            # set lookat point, start tracking is requested
            if selmode == 2 or selmode == 3:
                # set cam lookat
                if selbody >= 0:
                    self.cam.lookat = selpnt.flatten()
                # switch to tracking camera if dynamic body clicked
                if selmode == 3 and selbody > 0:
                    self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    self.cam.trackbodyid = selbody
                    self.cam.fixedcamid = -1
            # set body selection
            else:
                if selbody >= 0:
                    # record selection
                    self.pert.select = selbody
                    self.pert.skinselect = selskin
                    # compute localpos
                    vec = selpnt.flatten() - self.data.xpos[selbody]
                    mat = self.data.xmat[selbody].reshape(3, 3)
                    self.pert.localpos = self.data.xmat[selbody].reshape(3, 3).dot(vec)
                else:
                    self.pert.select = 0
                    self.pert.skinselect = -1
            # stop perturbation on select
            self.pert.active = 0

        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0,
                0.05 * y_offset,
                self.scn,
                self.cam,
            )

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)
