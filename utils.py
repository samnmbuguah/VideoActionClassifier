import av
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary of available models with their details
AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
    "VideoMAE Base": {
        "processor": "MCG-NJU/videomae-base",
        "model": "MCG-NJU/videomae-base",
        "description": "Base VideoMAE model for video understanding"
    },
    "VideoMAE Large": {
        "processor": "MCG-NJU/videomae-large",
        "model": "MCG-NJU/videomae-large",
        "description": "Large VideoMAE model with enhanced capabilities"
    }
}

def get_available_models() -> Dict[str, str]:
    """Return a dictionary of available models and their descriptions."""
    return {name: info["description"] for name, info in AVAILABLE_MODELS.items()}

def load_model(model_name: str = "VideoMAE Base") -> Tuple[VideoMAEImageProcessor, VideoMAEForVideoClassification]:
    """Load the video classification model and processor."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_name]
    try:
        processor = VideoMAEImageProcessor.from_pretrained(
            model_info["processor"],
            trust_remote_code=True
        )
        model = VideoMAEForVideoClassification.from_pretrained(
            model_info["model"],
            trust_remote_code=True
        )
        return processor, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

def resize_frame(frame: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Resize frame to target size while maintaining aspect ratio."""
    if not isinstance(frame, np.ndarray):
        raise ValueError("Input frame must be a numpy array")
    
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Invalid frame shape {frame.shape}. Expected (H, W, 3)")
        
    try:
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(pil_image)
    except Exception as e:
        logger.error(f"Error resizing frame: {str(e)}")
        raise ValueError(f"Failed to resize frame: {str(e)}")

def process_video(video_file, processor, model, num_frames=16):
    try:
        frames = []
        timestamps = []
        container = av.open(video_file)
        stream = container.streams.video[0]
        
        # Configure stream for performance
        stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
        stream.codec_context.thread_count = 8
        
        # Extract frames with temporal window consideration
        for frame in container.decode(video=0):
            frame_array = frame.to_ndarray(format="rgb24")
            resized_frame = resize_frame(frame_array, (224, 224))
            frames.append(resized_frame)
            timestamps.append(float(frame.pts * stream.time_base))
            
            if len(frames) >= num_frames:
                break
        
        # Ensure minimum number of frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
            timestamps.append(timestamps[-1])
        
        # Stack frames for temporal processing (B, C, T, H, W)
        frames_array = np.stack(frames)  # Shape: (T, H, W, C)
        frames_array = np.transpose(frames_array, (3, 0, 1, 2))  # Shape: (C, T, H, W)
        frames_array = np.expand_dims(frames_array, 0)  # Shape: (B, C, T, H, W)
        
        # Process temporal window
        try:
            inputs = processor(frames, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Process predictions
            scores = torch.nn.functional.softmax(logits, dim=1)[0]
            top_scores, top_indices = torch.topk(scores, k=min(5, len(model.config.id2label)))
            
            overall_results = [
                [(model.config.id2label[idx.item()], score.item())
                 for score, idx in zip(top_scores, top_indices)]
            ]
            
            # Process frame-level predictions
            frame_results = []
            for i, timestamp in enumerate(timestamps):
                frame_scores = torch.nn.functional.softmax(logits[:, :, i], dim=1)[0]
                frame_top_scores, frame_top_indices = torch.topk(frame_scores, k=min(3, len(model.config.id2label)))
                
                predictions = [
                    (model.config.id2label[idx.item()], score.item())
                    for score, idx in zip(frame_top_scores, frame_top_indices)
                ]
                
                frame_results.append({
                    'timestamp': timestamp,
                    'predictions': predictions
                })
            
            return overall_results, frame_results
            
        except Exception as e:
            logger.error(f"Error processing tensors: {str(e)}")
            raise RuntimeError(f"Failed to process video frames: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        raise RuntimeError(f"Failed to process video: {str(e)}")