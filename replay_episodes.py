import os
import h5py
import argparse
from collections import defaultdict 
from sim_env import make_sim_env
import matplotlib.pyplot as plt
import numpy as np

from utils import sample_box_pose, sample_insertion_pose
from sim_env import BOX_POSE
from constants import DT
from visualize_episodes import save_videos

import IPython
e = IPython.embed


def main(args):
    dataset_path = args['dataset_path']
    onscreen_render = True
    end_effector_pos = []


    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/observations/qpos'][()]

    env = make_sim_env('sim_RM_simpletrajectory')
    # BOX_POSE[0] = sample_box_pose() # used in sim reset
    ts = env.reset()
    episode_replay = [ts]

    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images']['top'])
        plt.ion()

    for action in actions:
        ts = env.step(action)
        episode_replay.append(ts)
        if onscreen_render:
            plt_img.set_data(ts.observation['images']['top'])
            plt.pause(0.02)
        end_effector = np.copy(ts.observation['position'])
        # print(f'end_effector: ', end_effector)
        end_effector_pos.append(end_effector)
        # print(f"Accumulated Positions: {end_effector_pos[-1:]}")  # 最近5个数据

    np.savetxt('/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/3.txt', np.array(end_effector_pos), fmt='%.6f', delimiter=',', comments='')
    # print(f"Accumulated Positions: {end_effector_pos[-5:]}")  # 最近5个数据
    print("End effector position has been saved! 'v'")

    # # saving
    # image_dict = defaultdict(lambda: [])
    # while episode_replay:
    #     ts = episode_replay.pop(0)
    #     for cam_name, image in ts.observation['images'].items():
    #         image_dict[cam_name].append(image)

    # video_path = dataset_path.replace('episode_', 'replay_episode_').replace('hdf5', 'mp4')
    # save_videos(image_dict, DT, video_path=video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', action='store', type=str, help='Dataset path.', required=True)
    main(vars(parser.parse_args()))
