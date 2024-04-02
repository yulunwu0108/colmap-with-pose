import numpy as np
import cv2 as cv
import os
from utils.read_write_model import rotmat2qvec
from utils.database import COLMAPDatabase
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dict_path', type=str, default='/mnt/disk1/wyl/datasets/DTU/scan24/cameras.npz')
parser.add_argument('--project_path', type=str, default='.')
parser.add_argument('--image_path', type=str, default='./images')

parser.add_argument('--n_images', type=int)
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
parser.add_argument('--case', type=str, default='')
parser.add_argument('--scaled', action='store_true')
args = parser.parse_args()

camera_dict = np.load(args.dict_path.replace('CASE_NAME', args.case))
n_images = args.n_images
w = args.width
h = args.height
scale_radius = 1.0

project_path = args.project_path.replace('CASE_NAME', args.case)
image_path = args.image_path.replace('CASE_NAME', args.case)
mask_path = image_path.replace('image', 'new_mask')
cam_path = image_path.replace('image', 'camera')

if os.path.exists(f'{project_path}/model'):
    os.system(f'rm -rf {project_path}/model')
if os.path.exists(f'{project_path}/sparse'):
    os.system(f'rm -rf {project_path}/sparse')
if os.path.exists(f'{project_path}/dense'):
    os.system(f'rm -rf {project_path}/dense')

os.makedirs(f'{project_path}/model', exist_ok=True)
os.makedirs(f'{project_path}/sparse', exist_ok=True)
os.makedirs(f'{project_path}/dense', exist_ok=True)

os.system(f'touch {project_path}/colmap_output.txt')
os.system(f'touch {project_path}/model/cameras.txt')
os.system(f'touch {project_path}/model/images.txt')
os.system(f'touch {project_path}/model/points3D.txt')


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # not R but R^-1
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = depth_min + float(lines[11].split()[1]) * 192
    depth_interval = float(lines[11].split()[1])
    intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
    intrinsics_[:3, :3] = intrinsics
    return intrinsics_, extrinsics, [depth_min, depth_max]


world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

intrinsics_all = []
pose_all = []

for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
    P = world_mat # @ scale_mat
    if args.scaled:
        P = P @ scale_mat
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)
    intrinsics_all.append(intrinsics)
    pose_all.append(pose)

data_list = []
for i in range(n_images):
    rt = pose_all[i]
    rt = np.linalg.inv(rt)
    r = rt[:3, :3]
    t = rt[:3, 3]
    # q = rotmat2qvec(r)
    q0 = 0.5 * np.sqrt(1 + r[0,0] + r[1,1] + r[2,2])
    q1 = (r[2,1] - r[1,2]) / (4 * q0)
    q2 = (r[0,2] - r[2,0]) / (4 * q0)
    q3 = (r[1,0] - r[0,1]) / (4 * q0)
    q = np.array([q0, q1, q2, q3])

    data = [i+1, *q, *t, i+1, '{:0>6d}.png'.format(i)] # {}.jpg
    data = [str(_) for _ in data]
    data = ' '.join(data)
    data_list.append(data)

with open(f'{project_path}/model/cameras.txt', 'w') as f:
    for i in range(n_images):
        f.write(f'{i+1} PINHOLE {w} {h} {intrinsics_all[i][0,0]} {intrinsics_all[i][1,1]} {intrinsics_all[i][0,2]} {intrinsics_all[i][1,2]}\n')

with open(f'{project_path}/model/images.txt', 'w') as f:
    for data in data_list:
        f.write(data)
        f.write('\n\n')

colmap_bin = 'colmap'

os.system(f'{colmap_bin} feature_extractor --database_path {project_path}/model/database.db --image_path {image_path} > {project_path}/colmap_output.txt') # --ImageReader.mask_path {mask_path}
os.system(f'python import.py --txtfile {project_path}/model/cameras.txt --database_path {project_path}/model/database.db >> {project_path}/colmap_output.txt')

os.system(f'{colmap_bin} exhaustive_matcher --database_path {project_path}/model/database.db >> {project_path}/colmap_output.txt')
db = COLMAPDatabase.connect(f'{project_path}/model/database.db')

images = list(db.execute('select * from images'))

data_list = []
for image in images:
    id = int(image[1][:-4])
    rt = pose_all[id]
    rt = np.linalg.inv(rt)
    r = rt[:3, :3]
    t = rt[:3, 3]
    q = rotmat2qvec(r)
    data = [image[0], *q, *t, id+1, '{:0>6d}.png'.format(id)] # {}.jpg
    data = [str(_) for _ in data]
    data = ' '.join(data)
    data_list.append(data)

with open(f'{project_path}/model/images.txt', 'w') as f:
    for data in data_list:
        f.write(data)
        f.write('\n\n')

os.system(f'{colmap_bin} point_triangulator --database_path {project_path}/model/database.db --image_path {image_path} --input_path {project_path}/model --output_path {project_path}/sparse >> {project_path}/colmap_output.txt') #--Mapper.tri_ignore_two_view_tracks 0
os.system(f'{colmap_bin} image_undistorter --image_path {image_path} --input_path {project_path}/sparse --output_path {project_path}/dense --output_type COLMAP >> {project_path}/colmap_output.txt')
os.system(f'{colmap_bin} patch_match_stereo --workspace_path {project_path}/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true >> {project_path}/colmap_output.txt')

ply_name = f'{args.case}.ply'
os.system(f'{colmap_bin} stereo_fusion --workspace_path {project_path}/dense --workspace_format COLMAP --input_type geometric --output_path {project_path}/dense/{ply_name} >> {project_path}/colmap_output.txt')

print('Reconstruction finished.')
