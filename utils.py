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

def process_segment(frames, timestamps, processor, model, num_frames=16):
    """Process a segment of video frames and return predictions."""
    if not frames or not timestamps:
        raise ValueError("Empty frames or timestamps")
    
    try:
        # Ensure we have enough frames
        if len(frames) < num_frames:
            # Duplicate last frame if needed
            frames.extend([frames[-1]] * (num_frames - len(frames)))
            timestamps.extend([timestamps[-1]] * (num_frames - len(timestamps)))
        elif len(frames) > num_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
            frames = [frames[i] for i in indices]
            timestamps = [timestamps[i] for i in indices]
        
        # Process frames in batches
        batch_frames = np.stack(frames)
        inputs = processor(list(batch_frames), return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get predictions
        scores = torch.nn.functional.softmax(logits, dim=1)[0]
        top_scores, top_indices = torch.topk(scores, k=5)
        
        overall_results = [
            (model.config.id2label[idx.item()], score.item())
            for score, idx in zip(top_scores, top_indices)
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
                'timestamp': float(timestamp),
                'predictions': predictions
            })
        
        return overall_results, frame_results
    except Exception as e:
        logger.error(f"Error in process_segment: {str(e)}")
        raise RuntimeError(f"Failed to process segment: {str(e)}")

def process_video_in_segments(video_file, processor, model, segment_duration=60):
    """Process video in segments to handle longer videos efficiently."""
    try:
        container = av.open(video_file)
        stream = container.streams.video[0]
        
        # Configure stream for better performance
        stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
        stream.codec_context.thread_count = 8
        
        # Get video metadata
        fps = float(stream.average_rate)
        duration = float(stream.duration * stream.time_base)
        frames_per_segment = int(fps * segment_duration)
        
        all_overall_results = []
        all_frame_results = []
        
        # Process video in segments
        for segment_idx in range(int(duration / segment_duration) + 1):
            segment_frames = []
            segment_timestamps = []
            
            # Seek to segment start
            start_time = segment_idx * segment_duration
            container.seek(int(start_time * stream.time_base * stream.rate))
            
            # Read frames for current segment
            for frame in container.decode(video=0):
                timestamp = frame.pts * float(stream.time_base)
                if timestamp >= (segment_idx + 1) * segment_duration:
                    break
                    
                try:
                    frame_array = frame.to_ndarray(format="rgb24")
                    resized_frame = resize_frame(frame_array, (224, 224))
                    segment_frames.append(resized_frame)
                    segment_timestamps.append(timestamp)
                except Exception as e:
                    logger.warning(f"Skipped frame at {timestamp}s: {str(e)}")
                    continue
            
            if segment_frames:
                try:
                    segment_overall, segment_frame_results = process_segment(
                        segment_frames,
                        segment_timestamps,
                        processor,
                        model
                    )
                    all_overall_results.append((start_time, segment_overall))
                    all_frame_results.extend(segment_frame_results)
                except Exception as e:
                    logger.error(f"Error processing segment {segment_idx}: {str(e)}")
        
        return all_overall_results, all_frame_results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise RuntimeError(f"Failed to process video: {str(e)}")

def process_video(video_file, processor, model, num_frames=16):
    """Process video file using segmented processing for improved handling of longer videos."""
    try:
        # Use segmented processing for all videos
        return process_video_in_segments(video_file, processor, model)
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        raise RuntimeError(f"Failed to process video: {str(e)}")
