import streamlit as st
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# Title and description
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image to find out if it's a cat or a dog!")

# API endpoint configuration
# You can set this via Streamlit secrets or environment variable
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a trained CNN model to classify images as cats or dogs.
    
    **How to use:**
    1. Upload an image (JPG, PNG)
    2. Wait for the prediction
    3. See the results!
    """)
    
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL, help="Enter your API endpoint URL")
    
    # Test API connection
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API is healthy!")
                st.json(response.json())
            else:
                st.error(f"❌ API returned status code: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {str(e)}")

# Main content
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a cat or dog"
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Prediction")
        
        # Make prediction button
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Prepare the file for upload
                    files = {"file": ("image.jpg", uploaded_file, "image/jpeg")}
                    
                    # Make API request
                    response = requests.post(
                        f"{api_url}/predict",
                        files=files,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display prediction
                        prediction = result["prediction"]
                        confidence = result["confidence"]
                        
                        # Show result with emoji
                        if prediction == "cat":
                            st.success(f"🐱 It's a **CAT**!")
                        else:
                            st.success(f"🐶 It's a **DOG**!")
                        
                        # Show confidence
                        st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Show probability breakdown
                        st.write("**Probability Breakdown:**")
                        probabilities = result["probabilities"]
                        
                        # Create progress bars
                        st.write(f"🐱 Cat: {probabilities['cat']:.1%}")
                        st.progress(probabilities['cat'])
                        
                        st.write(f"🐶 Dog: {probabilities['dog']:.1%}")
                        st.progress(probabilities['dog'])
                        
                    else:
                        st.error(f"Error: API returned status code {response.status_code}")
                        st.write(response.text)
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Please check the API URL in settings.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add some example images info
with st.expander("📸 Tips for best results"):
    st.write("""
    - Use clear, well-lit images
    - Ensure the cat or dog is the main subject
    - Avoid heavily filtered or edited images
    - Works best with close-up shots
    """)
