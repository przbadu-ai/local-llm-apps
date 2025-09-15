import tempfile
import streamlit as st
from lib.llm_config import LlmConfig
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Tuple, Optional
import re

llm_config = LlmConfig()

def extract_video_id(video_url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    # Handle different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)',
        r'youtube\.com\/live\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    
    # If no pattern matches, assume the input might be just the video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', video_url):
        return video_url
    
    return None

def fetch_video_data(video_url: str) -> Tuple[str, str, bool]:
    """
    Fetch video transcript from YouTube using the new API.
    
    Returns:
        Tuple of (video_id, transcript_text, success_flag)
    """
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return "", "Invalid YouTube URL format", False
        
        # Initialize the API
        ytt_api = YouTubeTranscriptApi()
        
        # Try to get transcript with various options
        try:
            # First, list all available transcripts
            transcript_list = ytt_api.list(video_id)
            
            # Try to find transcript in preferred order
            transcript = None
            
            try:
                # Try to get manually created English transcript first
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except:
                try:
                    # Fall back to auto-generated English transcript
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    try:
                        # Try to get any English transcript
                        transcript = transcript_list.find_transcript(['en'])
                    except:
                        # Get first available transcript in any language
                        for t in transcript_list:
                            transcript = t
                            break
            
            if transcript:
                # Fetch the actual transcript data
                fetched_transcript = transcript.fetch()
                
                # Extract text from the fetched transcript
                # The new API returns a FetchedTranscript object that's iterable
                transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
                
                return video_id, transcript_text, True
            else:
                return video_id, "No transcript could be retrieved", False
                
        except Exception as e:
            # Try direct fetch as fallback (will use default English)
            try:
                fetched_transcript = ytt_api.fetch(video_id)
                transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
                return video_id, transcript_text, True
            except Exception as fetch_error:
                return video_id, f"Error fetching transcript: {str(fetch_error)}", False
            
    except Exception as e:
        return "", f"Unexpected error: {str(e)}", False

# Streamlit app configuration
st.set_page_config(
    page_title="YouTube Video Chat Bot",
    page_icon="📺",
    layout="wide"
)

st.title("Chat with YouTube Video 📺")
st.caption("This app allows you to chat with a YouTube video using Ollama (llama3:instruct)")

openai_access_token = llm_config.api_key
if not openai_access_token and llm_config.provider == "openai":
    openai_access_token = st.text_input("OpenAI API Key", type="password")

# Initialize session state
if 'db_path' not in st.session_state:
    st.session_state.db_path = tempfile.mkdtemp()
    st.session_state.app = llm_config.create_bot(st.session_state.db_path, openai_access_token)
    st.session_state.video_added = False
    st.session_state.current_video_id = None

# Get the YouTube video URL from the user
col1, col2 = st.columns([3, 1])
with col1:
    video_url = st.text_input(
        "Enter YouTube Video URL", 
        placeholder="https://www.youtube.com/watch?v=... or just the video ID",
        help="Supports various YouTube URL formats including shorts and live videos"
    )
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    add_button = st.button("Add Video", type="primary", use_container_width=True)

# Add the video to the knowledge base
if video_url and add_button:
    with st.spinner("Fetching transcript..."):
        video_id, transcript, success = fetch_video_data(video_url)
        
        if success:
            try:
                # Check if this is a new video
                if video_id != st.session_state.current_video_id:
                    # Clear previous data and add new video
                    st.session_state.app.add(
                        transcript, 
                        data_type="text", 
                        metadata={"video_id": video_id, "url": video_url}
                    )
                    st.session_state.video_added = True
                    st.session_state.current_video_id = video_id
                    st.success(f"✅ Added video (ID: {video_id}) to knowledge base!")
                else:
                    st.info("This video is already in the knowledge base.")
                    
                # Display video info
                with st.expander("Transcript Preview", expanded=False):
                    preview_text = transcript[:1000] + "..." if len(transcript) > 1000 else transcript
                    st.text_area("Transcript", preview_text, height=200, disabled=True)
                    
            except Exception as e:
                st.error(f"Error adding video to knowledge base: {e}")
        else:
            st.error(f"❌ {transcript}")

# Chat interface
if st.session_state.video_added:
    st.divider()
    st.subheader("💬 Chat with the Video")
    
    # Create a form for better UX
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_area(
            "Ask any question about the YouTube Video",
            placeholder="E.g., What are the main points discussed in this video?",
            height=100
        )
        submit_button = st.form_submit_button("Send Question", type="primary")
    
    # Process the question
    if submit_button and prompt:
        with st.spinner("Thinking..."):
            try:
                # Get response from the bot
                response = st.session_state.app.chat(prompt)
                
                # Display Q&A
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.info(f"**Your Question:**\n{prompt}")
                with col2:
                    st.success(f"**Answer:**\n{response}")
                    
            except Exception as e:
                st.error(f"Error getting response: {e}")
else:
    if video_url:
        st.info("👆 Click 'Add Video' to load the transcript and start chatting!")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1. **Paste a YouTube URL** in the input field
    2. **Click 'Add Video'** to fetch the transcript
    3. **Ask questions** about the video content
    4. The AI will answer based on the video transcript
    
    **Supported URL formats:**
    - Standard: `youtube.com/watch?v=VIDEO_ID`
    - Short: `youtu.be/VIDEO_ID`
    - Shorts: `youtube.com/shorts/VIDEO_ID`
    - Live: `youtube.com/live/VIDEO_ID`
    - Just the video ID: `VIDEO_ID`
    
    **Note:** The video must have captions/transcripts available.
    """)
    
    st.divider()
    st.caption("Using Ollama with llama3:instruct and nomic-embed-text models")