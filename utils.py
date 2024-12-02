import av
import torch
import numpy as np
from typing import List
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

def load_model():
    """Load the video classification model and processor."""
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
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
