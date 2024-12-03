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
    "VideoMAE Kinetics": {
        "processor": "facebook/videomae-base-finetuned-kinetics",
        "model": "facebook/videomae-base-finetuned-kinetics",
        "description": "Trained on Kinetics-400 dataset for general action recognition"
    },
    "VideoMAE SSv2": {
        "processor": "facebook/videomae-base-finetuned-ssv2",
        "model": "facebook/videomae-base-finetuned-ssv2",
        "description": "Trained on Something-Something-V2 dataset for fine-grained action recognition"
    }
}

def get_available_models() -> Dict[str, str]:
    """Return a dictionary of available models and their descriptions."""
    return {name: info["description"] for name, info in AVAILABLE_MODELS.items()}

def load_model(model_name: str = "VideoMAE Kinetics") -> Tuple[VideoMAEImageProcessor, VideoMAEForVideoClassification]:
    """Load the video classification model and processor."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_name]
    processor = VideoMAEImageProcessor.from_pretrained(model_info["processor"])
    model = VideoMAEForVideoClassification.from_pretrained(model_info["model"])
    return processor, model

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

def validate_frame_dimensions(frame: np.ndarray) -> None:
    """Validate frame dimensions and format."""
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a numpy array")
    
    if len(frame.shape) != 3:
        raise ValueError(f"Invalid frame dimensions: expected 3 dimensions, got {len(frame.shape)}")
        
    if frame.shape[2] != 3:
        raise ValueError(f"Invalid color channels: expected 3 channels (RGB), got {frame.shape[2]}")

def process_video(video_file, processor, model, num_frames=16):
    try:
        frames = []
        timestamps = []
        container = av.open(video_file)
        stream = container.streams.video[0]
        
        # Configure stream for performance
        stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
        stream.codec_context.thread_count = 8
        
        # Calculate frame sampling interval
        frame_count = 0
        for frame in container.decode(video=0):
            frame_count += 1
        container.seek(0)
        
        interval = max(frame_count // num_frames, 1)
        
        # Extract frames with proper dimension handling
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx % interval == 0 and len(frames) < num_frames:
                # Ensure consistent frame dimensions
                frame_array = frame.to_ndarray(format="rgb24")
                resized_frame = resize_frame(frame_array, (224, 224))
                validate_frame_dimensions(resized_frame)
                frames.append(resized_frame)
                timestamps.append(float(frame.pts * stream.time_base))
        
        # Ensure we have exactly num_frames frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
            timestamps.append(timestamps[-1])
        
        frames = frames[:num_frames]  # Trim to exact number needed
        timestamps = timestamps[:num_frames]
        
        # Create input batch with proper dimensions
        try:
            inputs = processor(frames, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Process predictions
            scores = torch.nn.functional.softmax(logits, dim=1)[0]
            top_scores, top_indices = torch.topk(scores, k=5)
            
            overall_results = [
                [(model.config.id2label[idx.item()], score.item())
                 for score, idx in zip(top_scores, top_indices)]
            ]
            
            # Process individual frames
            frame_results = []
            for i, timestamp in enumerate(timestamps):
                frame_inputs = processor([frames[i]], return_tensors="pt")
                with torch.no_grad():
                    frame_outputs = model(**frame_inputs)
                    frame_logits = frame_outputs.logits
                
                frame_scores = torch.nn.functional.softmax(frame_logits, dim=1)[0]
                frame_top_scores, frame_top_indices = torch.topk(frame_scores, k=3)
                
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
