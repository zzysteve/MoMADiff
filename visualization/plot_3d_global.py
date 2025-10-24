import torch 
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from textwrap import wrap
import numpy as np
import torch
import io

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4, kinetic_chain=None, caption=None):
    joints, out_name, title = args

    data = joints.copy().reshape(len(joints), -1, 3)
    nb_joints = joints.shape[1]

    # 定义骨架连接
    if kinetic_chain is None:
        smpl_kinetic_chain = (
            [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20],
             [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
            if nb_joints == 21 else
            [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
             [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
             [9, 13, 16, 18, 20]]
        )
    else:
        smpl_kinetic_chain = kinetic_chain

    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    # 归一化高度 & 平移到根节点
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    # 创建 figure/axes
    fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=figsize, dpi=96)
    ax = fig.add_subplot(111, projection="3d")

    if title is not None:
        wraped_title = "\n".join(wrap(title, 40))
        fig.suptitle(wraped_title, fontsize=16)

    # 初始化坐标系
    def init_axes():
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(0, limits)
        ax.grid(False)
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    init_axes()

    # 画XZ地面平面
    def plot_xzPlane():
        verts = [
            [MINS[0], 0, MINS[2]],
            [MINS[0], 0, MAXS[2]],
            [MAXS[0], 0, MAXS[2]],
            [MAXS[0], 0, MINS[2]]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    plot_xzPlane()

    # 保存骨架线条对象
    skeleton_lines = []
    for _ in smpl_kinetic_chain:
        line, = ax.plot([], [], [], linewidth=2.0, color="black")
        skeleton_lines.append(line)

    # 保存轨迹线
    traj_line, = ax.plot([], [], [], linewidth=1.0, color="blue")

    # 每帧更新函数
    def update(index):
        # 更新轨迹
        if index > 1:
            traj_line.set_data(trajec[:index, 0] - trajec[index, 0],
                               np.zeros_like(trajec[:index, 0]))
            traj_line.set_3d_properties(trajec[:index, 1] - trajec[index, 1])

        # 更新骨架
        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            xs, ys, zs = data[index, chain, 0], data[index, chain, 1], data[index, chain, 2]
            skeleton_lines[i].set_data(xs, ys)
            skeleton_lines[i].set_3d_properties(zs)
            skeleton_lines[i].set_color(color)
            skeleton_lines[i].set_linewidth(4.0 if i < 5 else 2.0)

        return skeleton_lines + [traj_line]

    # 动画对象
    anim = FuncAnimation(fig, update, frames=frame_number, blit=False, interval=1000/fps)

    if caption is not None:
        fig.text(0.05, 0.02, caption, fontsize=12, color="black", ha="left", va="center", wrap=True)

    if out_name is not None:
        # 自动判断后缀保存 gif/mp4
        if out_name.endswith(".gif"):
            anim.save(out_name, writer="pillow", fps=fps)
        elif out_name.endswith(".mp4"):
            anim.save(out_name, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("Unsupported output format. Use .gif or .mp4")
        plt.close()
        return None
    else:
        # 返回 numpy 数组 (T, H, W, C)
        frames = []
        for i in range(frame_number):
            update(i)
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format="raw", dpi=96)
            io_buf.seek(0)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]),
                                       int(fig.bbox.bounds[2]),
                                       -1))
            frames.append(arr)
            io_buf.close()
        plt.close()
        return torch.from_numpy(np.stack(frames, axis=0))


# def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4, kinetic_chain=None, caption=None):
#     matplotlib.use('Agg')

#     joints, out_name, title = args

#     data = joints.copy().reshape(len(joints), -1, 3)

#     nb_joints = joints.shape[1]
#     if kinetic_chain is None:
#         smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
#     elif kinetic_chain:
#         smpl_kinetic_chain = kinetic_chain
#     limits = 1000 if nb_joints == 21 else 2
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)
#     colors = ['red', 'blue', 'black', 'red', 'blue',
#               'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
#               'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
#     frame_number = data.shape[0]
#     #     print(data.shape)

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     trajec = data[:, 0, [0, 2]]

#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     def update(index):

#         def init():
#             ax.set_xlim(-limits, limits)
#             ax.set_ylim(-limits, limits)
#             ax.set_zlim(0, limits)
#             ax.grid(b=False)
#         def plot_xzPlane(minx, maxx, miny, minz, maxz):
#             ## Plot a plane XZ
#             verts = [
#                 [minx, miny, minz],
#                 [minx, miny, maxz],
#                 [maxx, miny, maxz],
#                 [maxx, miny, minz]
#             ]
#             xz_plane = Poly3DCollection([verts])
#             xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#             ax.add_collection3d(xz_plane)
#         fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10), dpi=96)
#         if title is not None :
#             wraped_title = '\n'.join(wrap(title, 40))
#             fig.suptitle(wraped_title, fontsize=16)
#         ax = p3.Axes3D(fig)
        
#         init()
        
#         ax.lines = []
#         ax.collections = []
#         ax.view_init(elev=110, azim=-90)
#         ax.dist = 7.5
#         #         ax =
#         plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
#                      MAXS[2] - trajec[index, 1])
#         #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

#         if index > 1:
#             ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
#                       trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
#                       color='blue')
#         #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

#         for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
#             #             print(color)
#             if i < 5:
#                 linewidth = 4.0
#             else:
#                 linewidth = 2.0
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
#                       color=color)
#         #         print(trajec[:index, 0].shape)

#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

#         if caption is not None:
#             # TODO: put the caption in the bottom left corner
#             fig.text(0.05, 0.02, caption, fontsize=12, color='black', ha='left', va='center', wrap=True)
    
#         if out_name is not None : 
#             plt.savefig(out_name, dpi=96)
#             plt.close()
#         else : 
#             io_buf = io.BytesIO()
#             fig.savefig(io_buf, format='raw', dpi=96)
#             io_buf.seek(0)
#             # print(fig.bbox.bounds)
#             arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
#                                 newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
#             io_buf.close()
#             plt.close()
#             return arr

#     out = []
#     for i in range(frame_number) : 
#         out.append(update(i))
#     out = np.stack(out, axis=0)
#     return torch.from_numpy(out)


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None) : 
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size) : 
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help='motion npy file dir')
    parser.add_argument('--motion-list', default=None, nargs="+", type=str, help="motion name list")
    args = parser.parse_args()

    if args.motion_list is None:
        filename_list = glob(os.path.join(args.dir, '*.npy'))
        filename_list = [os.path.splitext(os.path.split(fpath)[-1])[0] for fpath in filename_list]
    else:
        filename_list = args.motion_list

    for filename in filename_list:
        motions = np.load(os.path.join(args.dir, f'{filename}.npy'))
        if motions.shape == 4:
            motions = motions[0]

        img = plot_3d_motion([motions, None, None])
        imageio.mimsave(os.path.join(args.dir, f'{filename}.gif'), np.array(img), fps=20)