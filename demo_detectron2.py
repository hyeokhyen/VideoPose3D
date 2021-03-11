import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

file_demo = '/nethome/hkwon64/Research/imuTube/repos_v2/pose/VideoPose3D/data/data_2d_custom_h36m_demo.npz'

keypoints = np.load(file_demo, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
print (keypoints_metadata)
assert False
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
keypoints = keypoints['positions_2d'].item()
print (keypoints.keys())

keypoints = keypoints['Walking.54138969.mp4']
keypoints = keypoints['custom']
keypoints = keypoints[0]
print (keypoints.shape)

for t in range(10):
  fig, ax = plt.subplots()
  kp_frame = keypoints[t]
  print ('kp_frame:', kp_frame.shape)
  for i in range(kp_frame.shape[0]):
    ax.plot(kp_frame[i,1], kp_frame[i,0], 'ro', markersize=10)
    ax.text(kp_frame[i,1], kp_frame[i,0], str(i), fontsize=15)
  ax.set_title('Detectron2: {}'.format(t))
  file_savefig = '/nethome/hkwon64/Research/imuTube/repos_v2/pose/VideoPose3D/demo/skeleton_idx_{}.png'.format(t)
  plt.savefig(file_savefig)
  print ('save in ...', file_savefig)
  plt.close()
