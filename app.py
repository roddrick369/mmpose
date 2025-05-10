import os
import sys
#import cv2

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
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer

def load_image(image_path):
    """Load image from a file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return mmcv.imread(image_path)

def initialize_detector(config_path, checkpoint_path, device='cuda:0'):
    """Initialize the object detector model."""
    detector = init_detector(config_path, checkpoint_path, device=device)
    return detector

def detect_and_visualize(detector, image_path, output_path, bbox_thr=0.3):
    """Detect objects and visualize bounding boxes."""
    # Load the image
    img = load_image(image_path)

    # Perform object detection
    results = inference_detector(detector, img)

    # Initialize the visualizer
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = detector.dataset_meta

    # Visualize results
    visualizer.add_datasample(
        name="result",
        image=img,
        data_sample=results,
        out_file=output_path,
        draw_pred=True,
        pred_score_thr=bbox_thr
    )

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Paths to config and checkpoint files
    det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
    det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

    # Input and output image paths
    input_image_path = '/content/image1.jpg'
    output_image_path = '/content/result.jpg'

    print("Initializing the detector...")
    detector = initialize_detector(det_config, det_checkpoint)

    print("Processing the image")
    detect_and_visualize(detector, input_image_path, output_image_path)

    print("Done!")