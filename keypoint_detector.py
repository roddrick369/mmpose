import os
import sys

print("Python usado:", sys.executable)
print("Versão:", sys.version)

def install_dependencies(mmcv_path=None):
    # Install required libraries using OpenMIM
    if mmcv_path:
        sys.path.insert(0, mmcv_path)
        os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{mmcv_path}"
        print(f"Usando mmcv no caminho especificado: {mmcv_path}")
    else:
        # Install foundational libraries
        os.system('python -m pip install -U pip')  # Upgrade pip
        os.system('python -m pip install openmim')  # Install OpenMIM
        #os.system('python -m mim install "mmcv>=2.0.0"')
        os.system('python -m mim install "mmengine==0.8.2"')
        os.system('python -m mim install "mmdet>=3.0.0"')

    # Install mmpose (editable mode for local development)
    os.system('python -m mim install -e .')

# Caminho para o mmcv
preinstalled_mmcv_path = '/content/drive/MyDrive/mmpose_cache'

print("Installing required dependencies...")
install_dependencies(preinstalled_mmcv_path)
print("Setup complete!")

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer

def load_image(image_path):
    """Load image from a file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return mmcv.imread(image_path)

def initialize_detector(config_path, checkpoint_path, device='cuda:0'):
    """Initialize the object detection model."""
    # Register all MMDetection modules
    register_all_modules()
    detector = init_detector(config_path, checkpoint_path, device=device)
    return detector

def initialize_pose_estimator(config_path, checkpoint_path, device='cuda:0'):
    """Initialize the pose estimation model."""
    pose_estimator = init_pose_estimator(config_path, checkpoint_path, device=device)
    return pose_estimator

def detect_and_visualize(detector, pose_estimator, image_path, output_path, bbox_thr=0.3, kpt_thr=0.3):
    """Detect objects, perform keypoint detection, and visualize results."""
    # Load the image
    img = load_image(image_path)

    # Perform object detection
    det_results = inference_detector(detector, img)
    bboxes = det_results.pred_instances.cpu().numpy()

    # Filter bounding boxes by score threshold
    valid_bboxes = bboxes[bboxes[:, -1] > bbox_thr]

    # Perform keypoint detection
    pose_results = inference_topdown(pose_estimator, img, valid_bboxes[:, :-1])
    data_samples = merge_data_samples(pose_results
                                      )
    # Initialize the visualizer
    det_visualizer = DetLocalVisualizer()
    det_visualizer.dataset_meta = detector.dataset_meta # Set dataset metadata

    # Visualize results
    det_img = det_visualizer.add_datasample(
        name="detection_result",
        image=img,
        data_sample=det_results,
        out_file=output_path,
        draw_pred=True,
        pred_score_thr=bbox_thr
    )

    # Overlay keypoints on the detection visualization
    pose_visualizer = DetLocalVisualizer()
    pose_visualizer.dataset_meta = pose_estimator.cfg.data.test.dataset_meta
    pose_visualizer.add_datasample(
        name="pose_result",
        image=det_img,
        data_sample=data_samples,
        draw_pred=True,
        pred_score_thr=kpt_thr,
        out_file=output_path
    )

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Paths to config and checkpoint files
    det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
    det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

    # Paths to pose estimator config and checkpoint files
    pose_config = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth'

    # Input and output image paths
    input_image_path = '/content/image1.jpg'
    output_image_path = '/content/result_pose.jpg'

    print("Initializing the detector...")
    pose_estimator = initialize_pose_estimator(pose_config, pose_checkpoint)
    detector = initialize_detector(det_config, det_checkpoint)

    print("Processing the image")
    detect_and_visualize(detector, pose_estimator, input_image_path, output_image_path)

    print("Done!")