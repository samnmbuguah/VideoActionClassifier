import av
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

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
    """Load the video classification model and processor.
    
    Args:
        model_name: Name of the model to load from available models
        
    Returns:
        Tuple of (processor, model)
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_name]
    processor = VideoMAEImageProcessor.from_pretrained(model_info["processor"])
    model = VideoMAEForVideoClassification.from_pretrained(model_info["model"])
    return processor, model

def process_video(video_file, processor, model, num_frames=16, frame_window=8):
    """Process video file and return both overall and frame-by-frame predictions.
    
    Args:
        video_file: Path to video file
        processor: VideoMAE processor
        model: VideoMAE model
        num_frames: Total number of frames to analyze
        frame_window: Number of frames to process in each window
        
    Returns:
        Tuple of (overall_results, frame_results)
    """
    # Read video using PyAV
    container = av.open(video_file)
    video_stream = container.streams.video[0]
    
    # Calculate sampling rate to get num_frames evenly spaced frames
    total_frames = video_stream.frames
    sampling_rate = max(total_frames // num_frames, 1)
    
    # Extract all required frames
    frames = []
    frame_timestamps = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % sampling_rate == 0 and len(frames) < num_frames:
            frames.append(frame.to_ndarray(format="rgb24"))
            frame_timestamps.append(frame.pts * float(video_stream.time_base))
    
    # If we don't have enough frames, duplicate the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
        frame_timestamps.append(frame_timestamps[-1] if frame_timestamps else 0)
    
    # Process all frames for overall prediction
    inputs = processor(frames, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Get overall predictions
    scores = torch.nn.functional.softmax(logits, dim=1)[0]
    top_scores, top_indices = torch.topk(scores, k=5)
    
    overall_results = []
    for score, idx in zip(top_scores, top_indices):
        label = model.config.id2label[idx.item()]
        overall_results.append((label, score.item()))
    
    # Process frame-by-frame using sliding window
    frame_results = []
    for i in range(0, len(frames) - frame_window + 1, frame_window // 2):
        window_frames = frames[i:i + frame_window]
        window_timestamp = frame_timestamps[i + frame_window // 2]  # Use middle frame timestamp
        
        # Process window
        window_inputs = processor(window_frames, return_tensors="pt")
        with torch.no_grad():
            window_outputs = model(**window_inputs)
            window_logits = window_outputs.logits
        
        # Get top 3 predictions for this window
        window_scores = torch.nn.functional.softmax(window_logits, dim=1)[0]
        window_top_scores, window_top_indices = torch.topk(window_scores, k=3)
        
        window_predictions = []
        for score, idx in zip(window_top_scores, window_top_indices):
            label = model.config.id2label[idx.item()]
            window_predictions.append((label, score.item()))
        
        frame_results.append({
            'timestamp': window_timestamp,
            'predictions': window_predictions
        })
    
    return overall_results, frame_results
