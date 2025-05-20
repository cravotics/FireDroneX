import cv2
import torch
import sys
import os
import numpy as np

# Add the Depth-Anything-V2 directory to sys.path
# Assuming this script is in a subdirectory of the workspace,
# and Depth-Anything-V2 is at the root of the workspace.
# Adjust the path if your Depth-Anything-V2 clone is elsewhere.
depth_anything_v2_repo_path = '/home/varun/ws_offboard_control/Depth-Anything-V2'
if depth_anything_v2_repo_path not in sys.path:
    sys.path.append(depth_anything_v2_repo_path)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    print(f"Failed to import DepthAnythingV2. Ensure '{depth_anything_v2_repo_path}' is correct, contains the 'depth_anything_v2' module, and that you have installed its requirements.")
    raise e

class DepthAnythingV2Estimator:
    def __init__(self, model_path, encoder='vits'):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"DepthAnythingV2Estimator using device: {self.DEVICE}")

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            # 'vitg' is for the giant model, not typically used with 'small' checkpoint
        }

        if encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {encoder}. Supported encoders are: {list(model_configs.keys())}")

        self.model = DepthAnythingV2(**model_configs[encoder])
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        except FileNotFoundError:
            print(f"Model checkpoint not found at {model_path}. Please ensure the path is correct.")
            raise
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise
            
        self.model = self.model.to(self.DEVICE).eval()
        print(f"Depth Anything V2 model (encoder: {encoder}) loaded from {model_path}")

    def predict_depth(self, image_bgr):
        """
        Predicts depth from a BGR image.
        Args:
            image_bgr: Input image in BGR format (OpenCV default).
        Returns:
            depth_map: HxW numpy array representing the depth map.
        """
        if image_bgr is None:
            print("Input image to predict_depth is None.")
            return None
        
        # The infer_image method from DepthAnythingV2 handles BGR to RGB conversion internally
        # and expects a BGR numpy array.
        # It also handles resizing to a square input_size (default 518) and then resizing back.
        try:
            depth_map = self.model.infer_image(image_bgr, input_size=518) # Using default input_size 518
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None
            
        return depth_map

if __name__ == '__main__':
    # Example Usage (for testing the estimator directly)
    print("Running DepthAnythingV2Estimator standalone test...")
    
    # This path should point to where you downloaded the .pth file
    # Example: /home/varun/ws_offboard_control/src/px4_ros_com/src/examples/depth_anything_models/depth_anything_v2_small.pth
    try:
        from ament_index_python.packages import get_package_share_directory
        # This assumes this test script is run in an environment where 'px4_ros_com' is a known package
        # This might not be true if you run `python depth_anything_v2_estimator.py` directly without sourcing a workspace
        package_share_dir = get_package_share_directory('px4_ros_com') # Adjust 'px4_ros_com' if package name is different for this test context
        model_checkpoint_path = os.path.join(package_share_dir, 'depth_anything_models', 'depth_anything_v2_vits.pth')
    except ImportError:
        print("ament_index_python not found or package not found. Using hardcoded relative path for model (may fail if not run from workspace src). ")
        # Fallback for environments where ament_index is not available or package not found, adjust as needed
        # This fallback is more for direct `python script.py` execution, less for `ros2 run`
        model_checkpoint_path = '../../examples/depth_anything_models/depth_anything_v2_vits.pth'
        # A better fallback might be an absolute path or an environment variable if standalone execution is important
        # For ROS 2 nodes, the get_package_share_directory method is preferred.

    # Create a dummy image for testing if no image is available locally
    test_image_path = 'test_image_cv.png'
    if not os.path.exists(test_image_path):
        print(f"Creating a dummy test image '{test_image_path}'")
        dummy_img_data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, dummy_img_data)

    try:
        print(f"Attempting to load model from: {model_checkpoint_path}")
        estimator = DepthAnythingV2Estimator(model_path=model_checkpoint_path, encoder='vits')
        
        print(f"Loading test image: {test_image_path}")
        img_bgr = cv2.imread(test_image_path) 
        
        if img_bgr is None:
            print(f"Failed to load {test_image_path}. Ensure the image exists or path is correct.")
        else:
            print(f"Test image loaded with shape: {img_bgr.shape}")
            depth = estimator.predict_depth(img_bgr)
            
            if depth is not None:
                print(f"Depth map predicted with shape: {depth.shape}, min_val: {depth.min():.2f}, max_val: {depth.max():.2f}")
                
                # Normalize for visualization (depth maps are often not in a displayable range directly)
                depth_normalized_for_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized_for_vis, cv2.COLORMAP_JET)
                
                output_depth_image_path = 'test_depth_map_v2_output.png'
                cv2.imwrite(output_depth_image_path, depth_colormap)
                print(f"Saved visualized depth map to '{output_depth_image_path}'")
            else:
                print("Depth prediction returned None.")
                
    except FileNotFoundError:
        print(f"CRITICAL: Model file not found at {model_checkpoint_path}. Please check the path.")
    except ImportError as e:
        print(f"CRITICAL: ImportError occurred: {e}. Check sys.path and Depth-Anything-V2 installation.")
    except Exception as e:
        print(f"An error occurred during the standalone test: {e}")
    
    # Clean up dummy image if created
    # if os.path.exists(test_image_path) and 'dummy_img_data' in locals():
    #     os.remove(test_image_path)
    #     print(f"Cleaned up dummy test image '{test_image_path}'") 