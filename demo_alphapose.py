import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from common.camera import *
from common.loss import *
from common.model import *
import torch
from common.visualization import render_animation
from moviepy.editor import VideoFileClip
from pprint import pprint

# Make sure to activate detectron2 environment
''' conda activate detectron2 '''
'''
cd inference
python infer_video_d2.py \
  --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
  --output-dir /coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo \
  --image-ext mp4 \
  /coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo
'''

'''
cd ../data
python prepare_data_2d_custom.py -i /coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo -o h36m_demo
'''

'''
cd ..
python run.py -d custom -k h36m_demo -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject Walking.54138969.mp4 --viz-action custom --viz-camera 0 --viz-video /coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo/Walking.54138969.mp4 --viz-output /coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo/Walking.54138969_output.mp4 --viz-size 6

'''

subject = 'S1'
action = 'Walking'
viz_camera = 0
pose_detector = 'AlphaPose' # AlphaPose | detectron2
is_render_trajectory = False
is_render_gt = False
coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

from common.h36m_dataset import Human36mDataset
dataset_path = 'data/data_3d_h36m.npz'
dataset = Human36mDataset(dataset_path)
# print (dataset.skeleton().num_joints())
cam = dataset.cameras()[subject][viz_camera]
pprint (cam)
assert False

file_video = '/nethome/hkwon64/Datasets/public/human36m/{subject}/Videos/{action}.54138969.mp4'.format(
  subject=subject, action=action)

# get video metadata for normalizing keypoints according to video resolution
# from inference.infer_video_d2 import get_resolution
# w, h = get_resolution(file_video)
# print ('w, h:', w, h)
clip = VideoFileClip(file_video)
w, h = clip.w, clip.h
coco_metadata['video_metadata'] = {'temp_name': {'h': h, 'w': w}}

keypoints_symmetry = coco_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])

joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

# load alpha pose results
if pose_detector == 'AlphaPose':
  file_ap = '/coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo/{pose_detector}/alphapose-results.json'.format(pose_detector=pose_detector)
  ap_results = json.load(open(file_ap))
  num_frames = len(ap_results)
  print ('num_frames:', num_frames)

  kps_ap = np.empty((num_frames, coco_metadata['num_joints'], 3))
  kps_ap[:] = np.nan
  for i in range(len(ap_results)):
    assert ap_results[i]['image_id'] == '{}.jpg'.format(i)
    kps_ap[i] = np.array(ap_results[i]['keypoints']).reshape((-1,3))
  assert np.all(np.isfinite(kps_ap))
  kps = kps_ap[:,:,:2]
  print ('kps:', kps.shape)
elif pose_detector == 'detectron2':
  file_demo = '/nethome/hkwon64/Research/imuTube/repos_v2/pose/VideoPose3D/data/data_2d_custom_h36m_demo.npz'

  keypoints = np.load(file_demo, allow_pickle=True)
  keypoints_metadata = keypoints['metadata'].item()
  keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
  #kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
  keypoints = keypoints['positions_2d'].item()
  #print (keypoints.keys())

  keypoints = keypoints['Walking.54138969.mp4'] # subject
  keypoints = keypoints['custom'] # action
  kps = keypoints[0] # cam_idx
  #print (kps.shape)

#print (dataset[subject][action].keys())
poses_3d = dataset[subject][action]['positions']
print ('poses_3d:', poses_3d.shape)
kps = kps[:poses_3d.shape[0]]
print ('kps:', kps.shape)

# Normalize camera frame
kps[..., :2] = normalize_screen_coordinates(kps[..., :2], 
  w=w, h=h)

# load pretrained model
architecture = '3,3,3,3,3'
filter_widths = [int(x) for x in architecture.split(',')]
causal = False
dropout = 0.25
channels = 1024
dense = False
model_pos = TemporalModel(
  coco_metadata['num_joints'], 
  2, 
  dataset.skeleton().num_joints(),
  filter_widths=filter_widths, 
  causal=causal, 
  dropout=dropout, 
  channels=channels,
  dense=dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
  model_pos = model_pos.cuda()

chk_filename = os.path.join('checkpoint', 'pretrained_h36m_detectron_coco.bin')
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
print('This model was trained for {} epochs'.format(checkpoint['epoch']))
model_pos.load_state_dict(checkpoint['model_pos'])

# prepare data for inference

with torch.no_grad():
  model_pos.eval()

  batch_2d = np.expand_dims(np.pad(kps[..., :2],
              ((pad + causal_shift, pad - causal_shift), 
              (0, 0), (0, 0)), 'edge'), axis=0)

  batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
  batch_2d[1, :, :, 0] *= -1
  batch_2d[1, :, kps_left + kps_right] = batch_2d[1, :, kps_right + kps_left]

  inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
  if torch.cuda.is_available():
      inputs_2d = inputs_2d.cuda()
  predicted_3d_pos = model_pos(inputs_2d)

  # Undo flipping and take average with non-flipped version
  predicted_3d_pos[1, :, :, 0] *= -1
  predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
  predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
  print ('predicted_3d_pos:', predicted_3d_pos.size())
  prediction = predicted_3d_pos.squeeze(0).cpu().numpy()

  # evaluate
  print('Preparing data...')
  for subject in dataset.subjects():
      for action in dataset[subject].keys():
          anim = dataset[subject][action]
          
          if 'positions' in anim:
              positions_3d = []
              for cam in anim['cameras']:
                  pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                  pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                  positions_3d.append(pos_3d)
              anim['positions_3d'] = positions_3d

  batch_3d = None if poses_3d is None else np.expand_dims(poses_3d, axis=0)
  batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
  print ('batch_3d:', batch_3d.shape)
  batch_3d[1, :, :, 0] *= -1
  batch_3d[1, :, joints_left + joints_right] = batch_3d[1, :, joints_right + joints_left]

  inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
  if torch.cuda.is_available():
      inputs_3d = inputs_3d.cuda()
  inputs_3d[:, :, 0] = 0    
  inputs_3d = inputs_3d[:1]

  epoch_loss_3d_pos_scale = inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

  error = mpjpe(predicted_3d_pos, inputs_3d)
  epoch_loss_3d_pos = inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
  N = inputs_3d.shape[0] * inputs_3d.shape[1]

  inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
  predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

  epoch_loss_3d_pos_procrustes = inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

  # Compute velocity error
  epoch_loss_3d_vel = inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

e1 = (epoch_loss_3d_pos / N)*1000
e2 = (epoch_loss_3d_pos_procrustes / N)*1000
e3 = (epoch_loss_3d_pos_scale / N)*1000
ev = (epoch_loss_3d_vel / N)*1000
print('Protocol #1 Error (MPJPE):', e1, 'mm')
print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
print('Velocity Error (MPJVE):', ev, 'mm')
print('----------')

# render
ground_truth = poses_3d.copy()

#print ('ground_truth:', ground_truth.shape)
#print ('prediction:', prediction.shape)

if is_render_trajectory:
  # Reapply trajectory
  trajectory = ground_truth[:, :1]
  ground_truth[:, 1:] += trajectory
  prediction += trajectory
cam = dataset.cameras()[subject][viz_camera]
pprint (cam)
assert False

prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
anim_output = {'Reconstruction': prediction}
input_keypoints = image_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
fps = 50
viz_bitrate = 3000
viz_limit = -1
viz_downsample = -1
viz_size = 6
viz_skip = 0
viz_video = '/coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo/Walking.54138969.mp4'
viz_output = '/coc/pcba1/hkwon64/imuTube/repos_v2/pose/VideoPose3D/demo/{pose_detector}/Walking.54138969_output'.format(pose_detector=pose_detector)
if not is_render_trajectory:
  viz_output += '_no_traj'

if is_render_gt:
  anim_output['Ground truth'] = ground_truth
else:
  viz_output += '_no_gt'
viz_output += '.mp4'
  
render_animation(input_keypoints, coco_metadata, anim_output,
                  dataset.skeleton(), fps, viz_bitrate, cam['azimuth'], viz_output,
                  limit=viz_limit, 
                  downsample=viz_downsample, 
                  size=viz_size,
                  input_video_path=viz_video, 
                  viewport=(cam['res_w'], cam['res_h']),
                  input_video_skip=viz_skip)
