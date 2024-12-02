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

def process_video(video_file, processor, model, num_frames=16):
    """Process video file and return predictions."""
    # Read video using PyAV
    container = av.open(video_file)
    video_stream = container.streams.video[0]
    
    # Calculate sampling rate to get num_frames evenly spaced frames
    total_frames = video_stream.frames
    sampling_rate = max(total_frames // num_frames, 1)
    
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % sampling_rate == 0 and len(frames) < num_frames:
            frames.append(frame.to_ndarray(format="rgb"))
    
    # If we don't have enough frames, duplicate the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    # Process frames with the model
    inputs = processor(frames, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Get predictions
    predicted_class_ids = logits.argmax(-1).item()
    predicted_labels = model.config.id2label[predicted_class_ids]
    
    # Get top 5 predictions with scores
    scores = torch.nn.functional.softmax(logits, dim=1)[0]
    top_scores, top_indices = torch.topk(scores, k=5)
    
    results = []
    for score, idx in zip(top_scores, top_indices):
        label = model.config.id2label[idx.item()]
        results.append((label, score.item()))
    
    return results
