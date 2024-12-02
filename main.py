import streamlit as st
import tempfile
from utils import load_model, process_video, get_available_models

# Page configuration
st.set_page_config(
    page_title="Video Action Classification",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("🎥 Video Action Classification")
st.markdown("""
Upload a video to identify the actions being performed. 
Select a model and the system will analyze the video to provide predictions with confidence scores.
""")

# Get available models
available_models = get_available_models()

# Add model selector
selected_model = st.selectbox(
    "Select Model",
    options=list(available_models.keys()),
    help="Choose the model to use for video classification"
)

# Display model description
st.info(f"**Model Details:** {available_models[selected_model]}")

# Initialize or update session state for model loading
if 'model' not in st.session_state or st.session_state.get('current_model') != selected_model:
    with st.spinner('Loading model... This may take a minute...'):
        st.session_state['processor'], st.session_state['model'] = load_model(selected_model)
        st.session_state['current_model'] = selected_model
    st.success('Model loaded successfully!')

# Show maximum file size warning
st.warning("Maximum file size: 200 MB. Larger files may cause upload errors.")

# File uploader with error handling
try:
    video_file = st.file_uploader(
        "Upload a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )

    if video_file is not None:
        # Check file size (200MB limit)
        file_size = len(video_file.getvalue()) / (1024 * 1024)  # Size in MB
        if file_size > 200:
            st.error(f"File size ({file_size:.1f} MB) exceeds the 200 MB limit. Please upload a smaller file.")
        else:
            # Display uploaded video
            st.video(video_file)
            
            try:
                # Create a temporary file to process the video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    # Process in chunks of 5MB
                    CHUNK_SIZE = 5 * 1024 * 1024
                    video_data = video_file.getvalue()
                    for i in range(0, len(video_data), CHUNK_SIZE):
                        chunk = video_data[i:i + CHUNK_SIZE]
                        tmp_file.write(chunk)
                    tmp_file_path = tmp_file.name
                
                # Process video with progress bar and segment status
                with st.spinner('Processing video...'):
                    overall_results, frame_results = process_video(
                        tmp_file_path,
                        st.session_state['processor'],
                        st.session_state['model']
                    )
                    
                    # Display results
                    st.success("Analysis complete!")
                    
                    # Display overall results by segment
                    st.subheader("Overall Video Analysis (By Segment):")
                    for segment_time, segment_results in overall_results:
                        with st.expander(f"Segment at {segment_time:.1f}s"):
                            for action, confidence in segment_results:
                                col1, col2, col3 = st.columns([2, 6, 2])
                                with col1:
                                    st.write(action)
                                with col2:
                                    st.progress(confidence)
                                with col3:
                                    st.write(f"{confidence*100:.1f}%")
                    
                    # Display frame-by-frame analysis
                    st.subheader("Frame-by-Frame Analysis:")
                    
                    # Create tabs for different visualization options
                    tab1, tab2 = st.tabs(["Temporal View", "Detailed Predictions"])
                    
                    with tab1:
                        # Create a temporal visualization of predictions
                        import altair as alt
                        import pandas as pd
                        
                        # Prepare data for visualization
                        temporal_data = []
                        for frame in frame_results:
                            for action, confidence in frame['predictions']:
                                temporal_data.append({
                                    'timestamp': frame['timestamp'],
                                    'action': action,
                                    'confidence': confidence
                                })
                        
                        df = pd.DataFrame(temporal_data)
                        df['timestamp'] = df['timestamp'].astype(float)
                        df['confidence'] = df['confidence'].astype(float)
                        df['action'] = df['action'].astype(str)
                        
                        # Create temporal chart with improved visualization
                        chart = alt.Chart(df).mark_line(
                            interpolate='linear',
                            point=True
                        ).encode(
                            x=alt.X('timestamp:Q', title='Time (seconds)'),
                            y=alt.Y('confidence:Q', title='Confidence Score', scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color('action:N', title='Action'),
                            tooltip=[
                                alt.Tooltip('action:N', title='Action'),
                                alt.Tooltip('confidence:Q', title='Confidence', format='.2%'),
                                alt.Tooltip('timestamp:Q', title='Time (s)', format='.2f')
                            ]
                        ).properties(
                            width=700,
                            height=400,
                            title='Action Confidence Over Time'
                        ).interactive()
                        
                        st.altair_chart(chart)
                    
                    with tab2:
                        # Display detailed frame-by-frame predictions with timeline
                        for i, frame in enumerate(frame_results):
                            with st.expander(f"Timestamp: {frame['timestamp']:.2f}s"):
                                for action, confidence in frame['predictions']:
                                    cols = st.columns([2, 6, 2])
                                    with cols[0]:
                                        st.write(action)
                                    with cols[1]:
                                        st.progress(confidence)
                                    with cols[2]:
                                        st.write(f"{confidence*100:.1f}%")
            
            except Exception as e:
                st.error(f"An error occurred while processing the video: {str(e)}")
                st.error("Please try uploading a different video file.")

except Exception as e:
    st.error(f"An error occurred during file upload: {str(e)}")

# Add footer with information
st.markdown("---")
st.markdown("""
### About
This application uses pre-trained video classification models from Hugging Face 
to identify actions in videos. Different models are available for various types
of action recognition tasks.

### Tips
- For best results, upload videos that are clear and well-lit
- The model works best with videos showing distinct actions
- Supported formats: MP4, AVI, MOV, MKV
- Longer videos are automatically processed in segments for better analysis
""")
