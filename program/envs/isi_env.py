from typing import Dict

from program.envs.isi_base_env import ISIBaseEnv


class ISIEnv(ISIBaseEnv):
    def __init__(self, initial_joint_qpos=None, **kwargs):
        ISIBaseEnv.__init__(self, initial_joint_qpos=initial_joint_qpos, **kwargs)

    def __get_info(self):
        info = dict()
        return info

    def _get_reset_info(self) -> Dict[str, float]:
        super_info = ISIBaseEnv._get_reset_info(self)
        info = self.__get_info()
        return super_info | info

    def reset_model(self):
        super_observation = ISIBaseEnv.reset_model(self)
        observation = super_observation
        return observation

    def step(self, action):
        super_observation, super_reward, super_terminated, super_truncated, super_info = ISIBaseEnv.step(
            self, action
        )
        info = self.__get_info()

        observation = super_observation
        reward = super_reward
        terminated = super_terminated
        truncated = super_truncated
        info = super_info | info

        return observation, reward, terminated, truncated, info
