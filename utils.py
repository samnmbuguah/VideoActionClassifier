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
        "processor": "MCG-NJU/videomae-base-finetuned-kinetics",
        "model": "MCG-NJU/videomae-base-finetuned-kinetics",
        "description": "Trained on Kinetics-400 dataset for general action recognition"
    },
    "VideoMAE SSv2": {
        "processor": "MCG-NJU/videomae-base-finetuned-ssv2",
        "model": "MCG-NJU/videomae-base-finetuned-ssv2",
        "description": "Trained on Something-Something-V2 dataset for fine-grained action recognition"
    },
    "VideoMAE UCF101": {
        "processor": "MCG-NJU/videomae-base-finetuned-ucf101",
        "model": "MCG-NJU/videomae-base-finetuned-ucf101",
        "description": "Trained on UCF101 dataset for sports and daily activities"
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

def process_video(video_file, processor, model, num_frames=16, frame_window=8):
    """Process video file and return both overall and frame-by-frame predictions."""
    frames = []
    frame_timestamps = []
    
    try:
        # Open video file with enhanced error handling
        container = av.open(video_file)
        stream = container.streams.video[0]
        stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
        stream.codec_context.thread_count = 8
        
        # Calculate frame interval for consistent sampling
        total_frames = stream.frames
        interval = max(total_frames // num_frames, 1)
        
        logger.info(f"Processing video: {total_frames} total frames, sampling interval: {interval}")
        logger.info(f"Video dimensions: {stream.width}x{stream.height}")
        
        # Extract frames with proper error handling
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx % interval == 0 and len(frames) < num_frames:
                try:
                    # Convert frame to RGB numpy array
                    frame_array = frame.to_ndarray(format="rgb24")
                    validate_frame_dimensions(frame_array)
                    
                    # Ensure frame is resized to model's expected input size
                    resized_frame = resize_frame(frame_array, (224, 224))
                    frames.append(resized_frame)
                    frame_timestamps.append(frame.pts * float(stream.time_base))
                    
                    logger.debug(f"Processed frame {frame_idx}: shape={resized_frame.shape}")
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_idx}: {str(e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading video file: {str(e)}")
        raise RuntimeError(f"Failed to process video file: {str(e)}")
    
    if not frames:
        raise ValueError("No valid frames could be extracted from the video")
    
    # Ensure we have enough frames by duplicating the last frame if necessary
    while len(frames) < num_frames:
        frames.append(frames[-1])
        frame_timestamps.append(frame_timestamps[-1] if frame_timestamps else 0)
    
    try:
        # Process frames for overall prediction
        logger.info(f"Processing {len(frames)} frames for overall prediction")
        inputs = processor(frames, return_tensors="pt")
        
        # Validate tensor dimensions
        if inputs['pixel_values'].shape != (1, num_frames, 3, 224, 224):
            raise ValueError(f"Invalid tensor shape: {inputs['pixel_values'].shape}")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get overall predictions
        scores = torch.nn.functional.softmax(logits, dim=1)[0]
        top_scores, top_indices = torch.topk(scores, k=5)
        
        overall_results = [
            (model.config.id2label[idx.item()], score.item())
            for score, idx in zip(top_scores, top_indices)
        ]
            
    except Exception as e:
        logger.error(f"Error in overall prediction: {str(e)}")
        raise RuntimeError(f"Failed to generate overall predictions: {str(e)}")
    
    # Process frame-by-frame using sliding window
    frame_results = []
    try:
        for i in range(0, len(frames) - frame_window + 1, frame_window // 2):
            window_frames = frames[i:i + frame_window]
            
            # Ensure consistent window size
            while len(window_frames) < frame_window:
                window_frames.append(window_frames[-1])
            
            window_timestamp = frame_timestamps[i + frame_window // 2]
            
            # Process window frames
            window_inputs = processor(window_frames, return_tensors="pt")
            
            # Validate tensor dimensions
            if window_inputs['pixel_values'].shape != (1, frame_window, 3, 224, 224):
                logger.warning(f"Invalid window tensor shape: {window_inputs['pixel_values'].shape}")
                continue
            
            with torch.no_grad():
                window_outputs = model(**window_inputs)
                window_logits = window_outputs.logits
            
            # Get predictions for this window
            window_scores = torch.nn.functional.softmax(window_logits, dim=1)[0]
            window_top_scores, window_top_indices = torch.topk(window_scores, k=3)
            
            window_predictions = [
                (model.config.id2label[idx.item()], score.item())
                for score, idx in zip(window_top_scores, window_top_indices)
            ]
            
            frame_results.append({
                'timestamp': window_timestamp,
                'predictions': window_predictions
            })
            
    except Exception as e:
        logger.error(f"Error in frame-by-frame analysis: {str(e)}")
        raise RuntimeError(f"Failed to generate frame-by-frame predictions: {str(e)}")
        
    return overall_results, frame_results
