from scipy.signal import savgol_filter  
import numpy as np
import os
from visualization.simplify_loc2rot import joints2smpl
import visualization.Animation as Animation

from visualization.InverseKinematics import BasicInverseKinematics, InverseKinematics
from visualization.Quaternions import Quaternions
import visualization.BVH_mod as BVH
import re
from visualization.remove_fs import *

def sanitize_filename(prompt):
    """
    Converts spaces to underscores and removes punctuation like commas and periods
    to generate a Linux-compliant filename.
    """
    # Replace spaces with underscores
    filename = prompt.replace(" ", "_")
    # Use regular expressions to remove punctuation
    filename = re.sub(r'[^\w\-]', '', filename)
    # Ensure the filename is not empty and conforms to general filename standards
    filename = filename.strip('_').lower()
    return filename

def filtering(joints, m_len, window_size=15, poly_order=3):
    pruned_data = joints[:m_len]

    T, V, C = pruned_data.shape
    # Transpose the data from (T, V, C) to (V, C, T) and reshape it to (V*C, T)
    pruned_data = pruned_data.transpose(1, 2, 0).reshape(-1, pruned_data.shape[0])

    data_smooth = savgol_filter(pruned_data, window_size, poly_order)

    # Convert back to (T, V, C)
    data_smooth = data_smooth.reshape(V, C, T).transpose(2, 0, 1)  # (T, V, C)

    return data_smooth

def convert_npy_to_smplx_npz(input_npy_path: str, output_npz_path: str, device: int = 0, cuda: bool = True):
    """
    A standalone function to directly convert a .npy file in (T, 22, 3) format
    to an SMPLX format .npz file.

    Args:
        input_npy_path (str): Path to the input .npy file (shape T, 22, 3).
        output_npz_path (str): Path to save the output .npz file.
        device (int): The GPU device ID to use.
        cuda (bool): Whether to use CUDA.
    """
    # --- load data ---
    if not os.path.exists(input_npy_path):
        print(f"Error: Input file does not exist at {input_npy_path}")
        return

    motion_data = np.load(input_npy_path, allow_pickle=True)
    num_frames, num_joints, _ = motion_data.shape
    
    if num_joints != 22:
        print(f"Warning: Expected 22 joints, but the file has {num_joints}. The result may be incorrect.")

    # --- 2. convert(T, V, C) to smpl ---
    print('Running SMPLify to fit SMPL model to joint data...')
    j2s = joints2smpl(num_frames=num_frames, device_id=device, cuda=cuda)
    motion_tensor, root_orient_tensor, trans_tensor, _, _ = j2s.joint2smpl(motion_data)
    
    pose_body_aa = motion_tensor.cpu().numpy()[0] # shape: (21, 3, T)
    root_orient = root_orient_tensor.cpu().numpy() # shape: (T, 3)
    trans = trans_tensor.cpu().numpy() # shape: (3, T)

    # --- 3. prepare parameters ---
    T = num_frames

    # convert (21, 3, T) to (T, 63)
    pose_body = pose_body_aa.transpose(2, 0, 1) # -> (T, 21, 3)
    pose_body = pose_body.reshape(T, -1)     # -> (T, 63)

    trans = trans.T

    pose_hand = np.zeros((T, 90))
    pose_jaw = np.zeros((T, 3))
    pose_eye = np.zeros((T, 6))
    
    poses = np.concatenate([
        root_orient,
        pose_body,
        pose_jaw,
        pose_eye,
        pose_hand
    ], axis=1) # (T, 165)

    betas = np.zeros(10) 
    num_betas = np.array(10, dtype=np.int32)
    gender = 'neutral'
    mocap_frame_rate = np.array(30.0, dtype=np.float32)
    mocap_time_length = np.array(num_frames / 30.0, dtype=np.float32)

    # --- 4. save as npz ---
    output_dir = output_npz_path
    print(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    filename = input_npy_path.split('/')[-1].split('.')[0]
    output_npz_path = os.path.join(output_dir, filename)
    print(output_npz_path)
        
    np.savez(
        output_npz_path,
        poses=poses,
        root_orient=root_orient,
        pose_body=pose_body,
        pose_hand=pose_hand,
        pose_jaw=pose_jaw,
        pose_eye=pose_eye,
        trans=trans,
        betas=betas,
        num_betas=num_betas,
        gender=gender,
        mocap_frame_rate=mocap_frame_rate,
        mocap_time_length = mocap_time_length
    )
    print(f"save to {output_npz_path}")


class Joint2BVHConvertor:
    def __init__(self):
        self.template = BVH.load('./visualization/data/template.bvh', need_quater=True)
        self.re_order = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]

        self.re_order_inv = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]
        self.end_points = [4, 8, 13, 17, 21]

        self.template_offset = self.template.offsets.copy()
        self.parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

    def convert(self, positions, filename, iterations=10, foot_ik=True):
        '''
        Convert the SMPL joint positions to Mocap BVH
        :param positions: (N, 22, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        '''
        positions = positions[:, self.re_order]
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(positions.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=-0)
        new_anim.positions[:, 0] = positions[:, 0]

        if foot_ik:
            positions = remove_fs(positions, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=5,
                                  force_on_floor=True)
        ik_solver = BasicInverseKinematics(new_anim, positions, iterations=iterations, silent=True)
        new_anim = ik_solver()

        glb = Animation.positions_global(new_anim)[:, self.re_order_inv]
        if filename is not None:
            BVH.save(filename, new_anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        return new_anim, glb
    def convert_sgd(self, positions, filename, iterations=100, foot_ik=True):
        '''
        Convert the SMPL joint positions to Mocap BVH

        :param positions: (N, 22, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        '''

        ## Positional Foot locking ##
        glb = positions[:, self.re_order]

        if foot_ik:
             glb = remove_fs(glb, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=2,
                                 force_on_floor=True)

        ## Fit BVH ##
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(glb.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(glb.shape[0], axis=-0)
        new_anim.positions[:, 0] = glb[:, 0]
        anim = new_anim.copy()

        rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
        pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
        offset = torch.tensor(anim.offsets, dtype=torch.float)

        glb = torch.tensor(glb, dtype=torch.float)
        ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)
        print('Fixing foot contact using IK...')
        for i in tqdm(range(iterations)):
            mse = ik_solver.step()
            # print(i, mse)

        rotations = ik_solver.rotations.detach().cpu()
        norm = torch.norm(rotations, dim=-1, keepdim=True)
        rotations /= norm

        anim.rotations = Quaternions(rotations.numpy())
        anim.rotations[:, self.end_points] = Quaternions.id((anim.rotations.shape[0], len(self.end_points)))
        anim.positions[:, 0, :] = ik_solver.position.detach().cpu().numpy()
        if filename is not None:
            BVH.save(filename, anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        # BVH.save(filename[:-3] + 'bvh', anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        glb = Animation.positions_global(anim)[:, self.re_order_inv]
        return anim, glb
