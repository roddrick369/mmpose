import os
import sys
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--input-type', choices=['image', 'video'], required=True, 
                    help="Specify input type: 'image' for a single image, 'video' for video processing.")
parser.add_argument('--input-path', type=str, required=True, help="Path to the input file (image or video).")
parser.add_argument('--output-path', type=str, required=True, help="Path to the output file (image or video).")

options = parser.parse_args()

print("Python usado:", sys.executable)
print("Versão:", sys.version)

class Args:
    def __init__(self):
        self.det_cat_id = 0  # Category ID for detection (e.g., 0 for person)
        self.bbox_thr = 0.3  # Bounding box score threshold
        self.nms_thr = 0.3   # IoU threshold for NMS
        self.kpt_thr = 0.3   # Keypoint score threshold
        self.radius = 6      # Keypoint radius
        self.alpha = 0.8     # Transparency of bounding boxes
        self.thickness = 3   # Skeleton line thickness

args = Args()

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

def load_image(image):
    """Load image from a file path or return it if already np.ndarray."""
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        return mmcv.imread(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise TypeError("Expected image to be file path or NumPy array.")

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

def detect_and_visualize(detector, pose_estimator, image_path, output_path, args):
    """Detect objects, perform keypoint detection, and visualize results."""
    from mmpose.evaluation.functional import nms

    # Load the image
    img = load_image(image_path)

    # Create a blank canvas with same dimensions as the input image
    canvas_height, canvas_width = img.shape[:2] # Get dimensions of the input image
    black_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Perform object detection
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()

     # Combine bounding boxes and scores
    if pred_instance.bboxes.size > 0 and pred_instance.scores.size > 0:
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    else:
        bboxes = np.array([])

    # Filter bounding boxes by class ID and score threshold
    if bboxes.size > 0:
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                       pred_instance.scores > args.bbox_thr)]

    # Apply Non-Maximum Suppression (NMS)
    if bboxes.size > 0:
        bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # Validate bounding boxes
    if bboxes is None or bboxes.size == 0:
        print("No valid bounding boxes found. Skipping pose estimation.")
        return


    # Convert valid_bboxes into the required format for inference_topdown
    #formatted_bboxes = [{'bbox': bbox.tolist()} for bbox in bboxes]

    # Perform keypoint detection
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # Initialize the visualizer
    det_visualizer = DetLocalVisualizer()
    det_visualizer.dataset_meta = detector.dataset_meta # Set dataset metadata

    # Visualize results
    det_img = det_visualizer.add_datasample(
        name="detection_result",
        image=img,
        data_sample=det_result,
        out_file=None,
        draw_pred=True,
        pred_score_thr=args.bbox_thr
    )

    # Initialize visualizer
    from mmpose.registry import VISUALIZERS

    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    # Set dataset metadata
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style='openpose')

    # Overlay keypoints on the detection visualization
    #pose_visualizer = DetLocalVisualizer()
    #pose_visualizer.dataset_meta = pose_estimator.cfg.data.test.dataset_meta

    # Add visualization
    visualizer.add_datasample(
        name="pose_result",
        image=black_canvas,
        data_sample=data_samples,
        draw_gt=False,
        draw_pred=True,
        out_file=output_path,
        skeleton_style="openpose"
    )

    print(f"Results saved to {output_path}")

    return data_samples.get('pred_instances', None)

def one_image_process(det, p_estimator, image_path, output_path, args):
    detect_and_visualize(det, p_estimator, image_path, output_path, args)

def video_process(det, p_estimator, input, output_path, args):
    import cv2
    from mmpose.visualization import FastVisualizer

    visualizer = FastVisualizer(
        p_estimator.dataset_meta,
        radius=args.radius,
        line_width=args.thickness,
        kpt_thr=args.kpt_thr
    )

    cap = cv2.VideoCapture(input)

    video_writer = None
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break

        pred_instances = detect_and_visualize(det, p_estimator, frame, output_path, args)

        visualizer.draw_pose(frame, pred_instances)

        frame_vis = frame.copy()[:, :, ::-1]
        
        output_file = 'test.mp4'
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_file,
                fourcc,
                25,
                (frame_vis.shape[1], frame_vis.shape[0])
            )
        video_writer.write(mmcv.rgb2bgr(frame_vis))
    video_writer.release()
    cap.release()
    return output_file

if __name__ == "__main__":
    # Paths to config and checkpoint files
    det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
    det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

    # Paths to pose estimator config and checkpoint files
    pose_config = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth'

    # Input and output image paths
    input_image_path = ''
    output_image_path = ''

    print("Initializing the detector...")
    pose_estimator = initialize_pose_estimator(pose_config, pose_checkpoint)
    detector = initialize_detector(det_config, det_checkpoint)

    if options.input_type == 'image':
        print("Processing the image")
        one_image_process(detector, pose_estimator, input_image_path, output_image_path, args)
    else:
        print("Processing the video")
        video_process(detector, pose_estimator, options.input_path, options.output_path, args)

    print("Done!")