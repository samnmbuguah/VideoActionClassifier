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

# File uploader
video_file = st.file_uploader(
    "Upload a video file", 
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Supported formats: MP4, AVI, MOV, MKV"
)

if video_file is not None:
    # Display uploaded video
    st.video(video_file)
    
    # Process button
    if st.button("Analyze Video"):
        try:
            # Create a temporary file to process the video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process video with progress bar
            with st.spinner('Processing video...'):
                results = process_video(
                    tmp_file_path,
                    st.session_state['processor'],
                    st.session_state['model']
                )
            
            # Display results
            st.success("Analysis complete!")
            
            # Create columns for better visualization
            st.subheader("Top 5 Predicted Actions:")
            
            # Display results in a nice format
            for action, confidence in results:
                col1, col2, col3 = st.columns([2, 6, 2])
                with col1:
                    st.write(action)
                with col2:
                    st.progress(confidence)
                with col3:
                    st.write(f"{confidence*100:.1f}%")
                    
        except Exception as e:
            st.error(f"An error occurred while processing the video: {str(e)}")
            st.error("Please try uploading a different video file.")

# Add footer with information
st.markdown("---")
st.markdown("""
### About
This application uses a pre-trained video classification model from Hugging Face 
to identify actions in videos. The model is trained on the Kinetics-400 dataset 
and can recognize various human actions.

### Tips
- For best results, upload videos that are clear and well-lit
- The model works best with videos showing distinct human actions
- Supported formats: MP4, AVI, MOV, MKV
""")
