def main():
    import os

    import psutil

    from internutopia.core.config import Config, SimConfig
    from internutopia.core.util import has_display
    from internutopia.core.vec_env import Env
    from internutopia.macros import gm
    from internutopia_extension import import_extensions
    from internutopia_extension.configs.robots.h1 import (
        H1RobotCfg,
        move_along_path_cfg,
        move_by_speed_cfg,
        rotate_cfg,
    )
    from internutopia_extension.configs.tasks import FiniteStepTaskCfg

    def memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)

    headless = False
    if not has_display():
        headless = True

    h1 = H1RobotCfg(
        position=(2.0, 0.0, 1.05),
        controllers=[
            move_by_speed_cfg,
            move_along_path_cfg,
            rotate_cfg,
        ],
        sensors=[],
    )

    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=True, rendering_interval=20, headless=headless
        ),
        env_num=1,
        env_offset_size=10,
        task_configs=[
            FiniteStepTaskCfg(
                max_steps=100,
                scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
                scene_scale=(0.01, 0.01, 0.01),
                robots=[h1],
            )
            for _ in range(100)
        ],
    )

    import_extensions()

    env = Env(config)
    obs, _ = env.reset()

    i = 0
    episode_idx = 0
    mem = None

    no_more_episode = False

    while env.simulation_app.is_running():
        obs, _, terminated_status, _, _ = env.step(action=[{}])

        if all(terminated_status) and no_more_episode:
            break

        if i % 100 == 0:
            print(i)

        if any(terminated_status) and not no_more_episode:
            obs, info = env.reset(env_ids=[idx for idx, term in enumerate(terminated_status) if term])
            episode_idx += 1

            if episode_idx % 20 == 0:
                _mem = memory_usage()
                if mem is not None:
                    assert (_mem - mem) < 40, f'memory leak detected during 20 episodes: {_mem-mem}MiB'
                mem = _mem
            if None in info:
                no_more_episode = True
        i += 1
    env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'exception is {e}')
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)
