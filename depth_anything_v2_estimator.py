import cv2
import torch
import sys
import os
import numpy as np

# Intializing the path to the Depth-Anything-V2 repository
depth_anything_v2_repo_path = '/home/varun/ws_offboard_control/Depth-Anything-V2'
if depth_anything_v2_repo_path not in sys.path:
    sys.path.append(depth_anything_v2_repo_path)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    print(f"Failed to import DepthAnythingV2. Ensure '{depth_anything_v2_repo_path}' is correct, contains the 'depth_anything_v2' module, and that you have installed its requirements.")
    raise e

# Creating a class for the DepthAnythingV2 Estimator incorporating the model and the depth prediction function.
class DepthAnythingV2Estimator:
    
    def __init__(self, model_path, encoder='vits'):
        # Selecting the CUDA or MPS device based on availability
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"DepthAnythingV2Estimator using device: {self.DEVICE}")
        
        # Defining the model configurations for the supported encoders
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        # Checking if the provided encoder is supported
        if encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {encoder}. Supported encoders are: {list(model_configs.keys())}")
        
        # Loading the model with the specified encoder and its configurations
        self.model = DepthAnythingV2(**model_configs[encoder])
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        except FileNotFoundError:
            print(f"Model checkpoint not found at {model_path}. Please ensure the path is correct.")
            raise
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise

        # Moving the model to the selected device and setting it to evaluation mode    
        self.model = self.model.to(self.DEVICE).eval()
        print(f"Depth Anything V2 model (encoder: {encoder}) loaded from {model_path}")
    
    def predict_depth(self, image_bgr):
        """
        Predicts depth from a BGR image.
        Args:
            image_bgr: Converts input Open CV image in BGR format.
        Returns:
            depth_map: depicts a numpy array of shape (H,W) representing the depth map.
        """
        if image_bgr is None:
            print("Input image to predict_depth is None.")
            return None
               

        try:
            # Performing the necessary preprocessing for the input image
            depth_map = self.model.infer_image(image_bgr, input_size=518) 
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None
            
        return depth_map

# Testing the DepthAnythingV2Estimator class
if __name__ == '__main__':
    print("Running DepthAnythingV2Estimator standalone test...")
    
    # Attempting to load the model checkpoint from the package share directory
    # or using a hardcoded relative path if the package is not found
    try:
        from ament_index_python.packages import get_package_share_directory
        package_share_dir = get_package_share_directory('px4_ros_com') 
        model_checkpoint_path = os.path.join(package_share_dir, 'depth_anything_models', 'depth_anything_v2_vits.pth')
    except ImportError:
        print("ament_index_python not found or package not found. Using hardcoded relative path for model (may fail if not run from workspace src). ")
        model_checkpoint_path = '../../examples/depth_anything_models/depth_anything_v2_vits.pth'

    # Creating a dummy image for testing if there is no image available locally
    test_image_path = 'test_image_cv.png'
    if not os.path.exists(test_image_path):
        print(f"Creating a dummy test image '{test_image_path}'")
        dummy_img_data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, dummy_img_data)

    try:
        # Loading the model and performing depth prediction
        print(f"Attempting to load model from: {model_checkpoint_path}")
        estimator = DepthAnythingV2Estimator(model_path=model_checkpoint_path, encoder='vits')
        
        # Loading the test image
        print(f"Loading test image: {test_image_path}")
        img_bgr = cv2.imread(test_image_path) 
        
        if img_bgr is None:
            print(f"Failed to load {test_image_path}. Ensure the image exists or path is correct.")
        else:
            print(f"Test image loaded with shape: {img_bgr.shape}")
            depth = estimator.predict_depth(img_bgr)
            
            if depth is not None:
                print(f"Depth map predicted with shape: {depth.shape}, min_val: {depth.min():.2f}, max_val: {depth.max():.2f}")
                
                # Normalization for Visualization
                depth_normalized_for_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized_for_vis, cv2.COLORMAP_JET)
                
                # Saving the visualized depth map
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
    
    