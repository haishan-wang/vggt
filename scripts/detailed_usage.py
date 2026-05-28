
import sys, os
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append( str(current_dir.parent) )

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
image_names = [f'examples/room/images/no_overlap_{i}.jpg' for i in range(2,9)]
images = load_and_preprocess_images(image_names).to(device)


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0], 
                                        [60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
    
print('Input images', images.shape)
print('Pose_enc', pose_enc.shape)
print('Extrinsic', extrinsic.shape)
print('Intrinsic', intrinsic.shape)
print('Depth_map', depth_map.shape)
print('Depth_conf', depth_conf.shape)
print('Point_map', point_map.shape)
print('Point_conf', point_conf.shape)
print('Point_map_by_unprojection', point_map_by_unprojection.shape)
# print('Track_list', track_list)
print('Vis_score', vis_score.shape)
print('Conf_score', conf_score.shape)

breakpoint()
