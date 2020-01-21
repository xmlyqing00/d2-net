

import argparse

import imagesize

import numpy as np

import os

parser = argparse.ArgumentParser(description='MegaDepth preprocessing script')

parser.add_argument(
    '--base_path', type=str, required=True,
    help='path to MegaDepth'
)
parser.add_argument(
    '--scene_id', type=str, required=True,
    help='scene ID'
)

parser.add_argument(
    '--output_path', type=str, required=True,
    help='path to the output directory'
)

args = parser.parse_args()

base_path = args.base_path
# Remove the trailing / if need be.
if base_path[-1] in ['/', '\\']:
    base_path = base_path[: - 1]
scene_id = args.scene_id

base_depth_path = os.path.join(
    base_path, 'phoenix/S6/zl548/MegaDepth_v1'
)
base_undistorted_sfm_path = os.path.join(
    base_path, 'Undistorted_SfM'
)

# sparse-txt 文件夹下有三个 txt 文件, 分别是 cameras, images 和 points3D
# cameras
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
#   相机ID | 相机模型(大多都是针孔模型) | 相机拍摄得到的图像的宽和高 | 相机的4个内参(f_x, f_y, u, v)
# images
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
#   图像ID | QW, QX, QY, QZ 是四元组, 可以根据它计算得到相机外参的R | TX, TY, TZ 外参的T | 相机ID | 图像文件名
#   该图像ID上若干个配准点, 分别用像素坐标 X Y 和 对应的三维点坐标 POINT3D_ID 表示, 如果 POINT3D_ID 是 -1, 则表示该匹配点没有对应的 3D 点
# points3D
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
#   3D点的ID | 点的 X Y Z 坐标 | 该点的 RGB 值? | 误差? | 和一系列的图像二维点, 分别用 图像ID 和 POINTS2D[]的下标表示, 下标从 0 开始
undistorted_sparse_path = os.path.join(
    base_undistorted_sfm_path, scene_id, 'sparse-txt'
)
if not os.path.exists(undistorted_sparse_path):
    exit()

depths_path = os.path.join( # 图像的深度图, 大小和图像是一样大的
    base_depth_path, scene_id, 'dense0', 'depths'
)
if not os.path.exists(depths_path):
    exit()

images_path = os.path.join(
    base_undistorted_sfm_path, scene_id, 'images'
)
if not os.path.exists(images_path):
    exit()

# Process cameras.txt
with open(os.path.join(undistorted_sparse_path, 'cameras.txt'), 'r') as f:
    raw = f.readlines()[3 :]  # skip the header # 跳过 cameras.txt 中对于格式介绍的前 3 行 # len(raw) = 3803

camera_intrinsics = {}
for camera in raw: # 读取一行的数据 CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    camera = camera.split(' ') # 以空格分隔
    camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2 :]] # 将 WIDTH, HEIGHT, PARAMS[] 转换为 list 存入以 CAMERA_ID 为索引的字典 camera_intrinsics 中

# Process points3D.txt
with open(os.path.join(undistorted_sparse_path, 'points3D.txt'), 'r') as f:
    raw = f.readlines()[3 :]  # skip the header # 同样跳过文件开头3行对于数据格式的介绍 # len(raw) = 225128

points3D = {}
for point3D in raw:
    point3D = point3D.split(' ') # 读取一行, 同样以空格分隔, 包含 POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    points3D[int(point3D[0])] = np.array([ # 将 points3D 的 X, Y, Z 世界坐标存入以 point3D_id 为索引的字典 points3D 中
        float(point3D[1]), float(point3D[2]), float(point3D[3])
    ])
    
# Process images.txt
with open(os.path.join(undistorted_sparse_path, 'images.txt'), 'r') as f:
    raw = f.readlines()[4 :]  # skip the header # # 同样跳过文件开头4行对于数据格式的介绍 # len(raw) = 7606

image_id_to_idx = {} # 记录 IMAGE_ID 在 images.txt 中对应下标 index 的字典
image_names = [] # 保存 image name, 可以通过 idx 索引, idx 可以通过 image_id_to_idx 索引, 所以知道了 IMAGE_ID 就可以知道 image_name
raw_pose = [] # 保存拍摄该图像时的相机外参(R与T)
camera = [] # 保存 camera id
points3D_id_to_2D = [] # 保存每张图像里重建 SfM 的POINT3D_ID到二维点的映射
n_points3D = [] # 记录每个图像匹配的三维点的个数
for idx, (image, points) in enumerate(zip(raw[:: 2], raw[1 :: 2])): # 因为 cameras.txt 是每两行存储了 image 的信息, 所以这里使用 zip 将两行合并为一行
    image = image.split(' ') # 以空格分割 IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    points = points.split(' ') # 以空格分隔 POINTS2D[] as (X, Y, POINT3D_ID)

    image_id_to_idx[int(image[0])] = idx # 记录 IMAGE_ID 对应的 idx

    image_name = image[-1].strip('\n') # 诸如 5978745_520771b1cb_o.jpg
    image_names.append(image_name)

    raw_pose.append([float(elem) for elem in image[1 : -2]]) # image[1 : -2] 为 QW, QX, QY, QZ, TX, TY, TZ # 保存拍摄该图像时的相机外参(R与T)
    camera.append(int(image[-2])) # 保存 CAMERA_ID
    current_points3D_id_to_2D = {} # 保存 POINT3D_ID 到 像素坐标的映射 # 3d_id to 2d (x,y) pixel coordinator in image_id
    for x, y, point3D_id in zip(points[:: 3], points[1 :: 3], points[2 :: 3]):
        if int(point3D_id) == -1:
            continue
        current_points3D_id_to_2D[int(point3D_id)] = [float(x), float(y)]
    points3D_id_to_2D.append(current_points3D_id_to_2D)
    n_points3D.append(len(current_points3D_id_to_2D)) # 记录当前图像匹配的三维点的个数
n_images = len(image_names) # 记录该 scene 下的照片数量

# Image and depthmaps paths
image_paths = [] # 记录图像的相对路径
depth_paths = [] # 记录深度图的相对路径
for image_name in image_names:
    image_path = os.path.join(images_path, image_name) #
   
    # Path to the depth file
    depth_path = os.path.join(
        depths_path, '%s.h5' % os.path.splitext(image_name)[0]
    )
    
    if os.path.exists(depth_path): # 有可能某些图像没有对应的深度图
        # Check if depth map or background / foreground mask
        file_size = os.stat(depth_path).st_size # 读取的单位是 Byte, 转为 KBytes 需要除以 1024
        # Rough estimate - 75KB might work as well
        if file_size < 100 * 1024: # 如果小于 100 KB 则不用这个数据
            depth_paths.append(None)
            image_paths.append(None)
        else:
            depth_paths.append(depth_path[len(base_path) + 1 :]) # 保存去掉 base_path 的相对路径
            image_paths.append(image_path[len(base_path) + 1 :]) # 保存去掉 base_path 的相对路径
    else:
        depth_paths.append(None)
        image_paths.append(None)

# Camera configuration
intrinsics = [] # 相机内参
poses = [] # 相机外参
principal_axis = [] # Z轴旋转参数
points3D_id_to_ndepth = [] # POINT3D_ID 在图像上的深度
for idx, image_name in enumerate(image_names):
    if image_paths[idx] is None: # 为 None 就是有可能是 深度图不存在 或者 深度图 size 小于 100 KB
        intrinsics.append(None)
        poses.append(None)
        principal_axis.append([0, 0, 0])
        points3D_id_to_ndepth.append({})
        continue
    image_intrinsics = camera_intrinsics[camera[idx]] # camera[idx] 可以取得对应的 CAMERA_ID, 然后可以从字典 camera_intrinsics 中取得 WIDTH, HEIGHT, PARAMS[]
    K = np.zeros([3, 3])
    K[0, 0] = image_intrinsics[2] # f_x
    K[0, 2] = image_intrinsics[4] # u
    K[1, 1] = image_intrinsics[3] # f_y
    K[1, 2] = image_intrinsics[5] # v
    K[2, 2] = 1
    intrinsics.append(K) # 保存相机内参

    image_pose = raw_pose[idx] # raw_pose 保存的是 QW, QX, QY, QZ, TX, TY, TZ
    qvec = image_pose[: 4] # 四元组
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([ # 将四元组转换为相机在拍摄该图像 image_name 时的旋转矩阵 R
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w
        ],
        [
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w
        ],
        [
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y
        ]
    ])
    principal_axis.append(R[2, :]) # 对于 Z 轴旋转的参数
    t = image_pose[4 : 7] # 外参中的平移参数 T
    # World-to-Camera pose # 世界坐标系到相机坐标系的 4x4 变换矩阵
    current_pose = np.zeros([4, 4])
    current_pose[: 3, : 3] = R
    current_pose[: 3, 3] = t
    current_pose[3, 3] = 1
    # Camera-to-World pose # 我们同样可以推导出对应的相机坐标系到世界坐标系的 4x4 矩阵
    # pose = np.zeros([4, 4])
    # pose[: 3, : 3] = np.transpose(R)
    # pose[: 3, 3] = -np.matmul(np.transpose(R), t)
    # pose[3, 3] = 1
    poses.append(current_pose)
    
    current_points3D_id_to_ndepth = {} # 保存图像 image_name 中所有 points3D, 以 POINT3D_ID 为索引到该三维点深度的字典
    for point3D_id in points3D_id_to_2D[idx].keys():
        p3d = points3D[point3D_id] # 取得世界坐标系
        # average focal lengths
        # ref: [2002] CAMERA CALIBRATION METHOD FOR STEREO MEASUREMENTS - page 6
        ave_f = .5 * (K[0, 0] + K[1, 1]) # 平均 f_x 与 f_y # average focal lengths(焦距)
        current_points3D_id_to_ndepth[point3D_id] = (np.dot(R[2, :], p3d) + t[2]) / ave_f # (np.dot(R[2, :], p3d) + t[2]) 得到 相机坐标系 下的 Z_camera # 再除以焦距就得到了图像深度 # 以 POINT3D_ID 为索引记录所有 POINT3D 的在该图像的图像深度
    points3D_id_to_ndepth.append(current_points3D_id_to_ndepth) # 保存该图像上所有用于重建 SfM 的 POINT3D 的在该图像的图像深度
principal_axis = np.array(principal_axis) # 相机外参对于 Z 轴旋转的参数
angles = np.rad2deg(np.arccos( # 这个不太清楚时什么, 在 D2Net 也没有用到
    np.clip(
        np.dot(principal_axis, np.transpose(principal_axis)),
        -1, 1
    )
))

# Compute overlap score
overlap_matrix = np.full([n_images, n_images], -1.) # 该场景下的照片两两间的重合率(两张照片 POINT3D 的重复率)
scale_ratio_matrix = np.full([n_images, n_images], -1.) # 两张照片之间重合 POINT3D  的最小尺度变化率
for idx1 in range(n_images): # 从 0, 1, 2 到 n_images 遍历
    if image_paths[idx1] is None or depth_paths[idx1] is None:
        continue
    for idx2 in range(idx1 + 1, n_images): # 根据 idx1 加 1 之后 从 1, 2, 3 到 n_images 遍历, 和冒泡排序差不多
        if image_paths[idx2] is None or depth_paths[idx2] is None:
            continue
        matches = ( # 通过 & 运算 找到相同 POINT3D_ID, & 运算以后会返回一个 set # matches = {378053, 388807, 468569, 476922, 597446}, 是一个 set
            points3D_id_to_2D[idx1].keys() & # points3D_id_to_2D[idx1].keys() 是 POINT3D_ID 代表图1 上的所有 POINT3D
            points3D_id_to_2D[idx2].keys()
        )
        min_num_points3D = min( # 这个数据没有用到
            len(points3D_id_to_2D[idx1]), len(points3D_id_to_2D[idx2])
        )
        overlap_matrix[idx1, idx2] = len(matches) / len(points3D_id_to_2D[idx1])  # 计算图1到图2的POINT3D重复率
        overlap_matrix[idx2, idx1] = len(matches) / len(points3D_id_to_2D[idx2])  # 计算图2到图1的POINT3D重复率
        if len(matches) == 0:
            continue
        points3D_id_to_ndepth1 = points3D_id_to_ndepth[idx1]
        points3D_id_to_ndepth2 = points3D_id_to_ndepth[idx2]
        nd1 = np.array([points3D_id_to_ndepth1[match] for match in matches]) # 取出重合的所有 POINT3D 在图像1上的深度
        nd2 = np.array([points3D_id_to_ndepth2[match] for match in matches]) # 取出重合的所有 POINT3D 在图像2上的深度
        min_scale_ratio = np.min(np.maximum(nd1 / nd2, nd2 / nd1)) # 先取出 np.maximum(nd1 / nd2, nd2 / nd1) 大于 1 大那个 比值 # 然后取所有 POINT3D 中尺度最小的比值
        scale_ratio_matrix[idx1, idx2] = min_scale_ratio
        scale_ratio_matrix[idx2, idx1] = min_scale_ratio

np.savez( # 保存这些所有的数据
    os.path.join(args.output_path, '%s.npz' % scene_id),
    image_paths=image_paths,
    depth_paths=depth_paths,
    intrinsics=intrinsics,
    poses=poses,
    overlap_matrix=overlap_matrix,
    scale_ratio_matrix=scale_ratio_matrix,
    angles=angles,
    n_points3D=n_points3D,
    points3D_id_to_2D=points3D_id_to_2D,
    points3D_id_to_ndepth=points3D_id_to_ndepth
)
