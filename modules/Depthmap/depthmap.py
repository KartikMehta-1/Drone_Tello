import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image  # Import PIL for conversion

# Function to load the MiDaS model
def load_midas_model():
    model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
    model_path = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    model = model_path.eval()  # Set to evaluation mode
    return model


# Preprocess input for the MiDaS model
def preprocess(image):
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(image)

    # Define transformations
    transform = Compose(
        [
            Resize((384, 384)),  # Resize to MiDaS input size
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(pil_image).unsqueeze(0)


# Post-process the depth map for visualization
def postprocess(depth_map):
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map = (depth_map * 255).astype(np.uint8)
    return depth_map


# Main function to capture depth from webcam
def main():
    # Load MiDaS model
    print("Loading MiDaS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_midas_model().to(device)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert BGR to RGB and preprocess the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(frame_rgb).to(device)  # Send input to GPU

        # Predict depth
        # Predict depth
        with torch.no_grad():
            depth_map = model(input_tensor)  # Model runs on GPU

            # Add a channel dimension to match (N, C, H, W) format
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),  # Add channel dimension
                size=frame.shape[:2],  # Match the frame size (H, W)
                mode="bicubic",
                align_corners=False
            ).squeeze(1)  # Remove channel dimension for processing

        # Post-process and display the depth map
        depth_map = postprocess(depth_map)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)  # Add color map

        # Show the original frame and depth map
        cv2.imshow("Webcam Feed", frame)
        cv2.imshow("Depth Map", depth_map)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
