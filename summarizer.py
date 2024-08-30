#dependencies
import fitz
from PIL import Image
import pytesseract
import re
import string
from transformers import BartTokenizer, BartForConditionalGeneration
import streamlit as st
import tempfile
import pathlib
import speech_recognition as sr
from pytube import YouTube
import requests
from moviepy.editor import VideoFileClip

####################################################################

def summarize_text(text):
    # Load pre-trained BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and generate summary
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Calculate max and min length based on the input text length
    max_length = len(inputs[0]) // 3
    min_length = len(inputs[0]) // 4

    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    try:
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except:
        summary = "Error while summarizing"
    return summary

####################################################################

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

####################################################################

# get text from the image
def img2text(image_path):
    original_image = Image.open(image_path)
    
    # Getting more good results if we are dividing the images 
    
    width, height = original_image.size
    part_height = height // 3

    # Create empty images for each part
    part1 = Image.new('RGB', (width, part_height))
    part2 = Image.new('RGB', (width, part_height))
    part3 = Image.new('RGB', (width, part_height))

    # Crop the original image into three equal parts
    part1_data = original_image.crop((0, 0, width, part_height))
    part2_data = original_image.crop((0, part_height, width, 2 * part_height))
    part3_data = original_image.crop((0, 2 * part_height, width, height))

    # Save the three parts as separate images with unique filenames
    part1_data.save('downloaded/part1.png')
    part2_data.save('downloaded/part2.png')
    part3_data.save('downloaded/part3.png')

    # Extract text from each part
    custom_config = r'--oem 3 --psm 6 --dpi 300 -l eng'

    text1 = pytesseract.image_to_string('downloaded/part1.png', config=custom_config)
    text2 = pytesseract.image_to_string('downloaded/part2.png', config=custom_config)
    text3 = pytesseract.image_to_string('downloaded/part3.png', config=custom_config)

    combined_text = text1 + text2 + text3
    
    return preprocess_text(combined_text)

####################################################################

# create img of ecah pdf and extract text
def pdf_img2text(doc, page_number):
    page = doc.load_page(page_number)
    pix = page.get_pixmap()  # Render the page as an image in memory
    
    # Create a PIL image from the raw data
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save('img.png')
    # Extract text from the image
    temp_text = img2text('img.png')
    
    return temp_text

####################################################################

# full pdf to text
def pdf2text(image_path):
    doc = fitz.open(image_path)
    total_pages = doc.page_count
    text = ""
    for i in range(total_pages):
        page_number = i
        
        try:
            temp_text = pdf_img2text(doc, page_number)
            text = text + temp_text
        except Exception as e:
            print(f"Error processing page {page_number}: {e}")
            
    doc.close()
    
    return preprocess_text(text)

####################################################################

def audio2text(audio_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    # Load the audio file using the PocketSphinx recognizer
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        
    try:
        text = recognizer.recognize_sphinx(audio_data)
    except sr.UnknownValueError:
        text = "Sphinx could not understand the audio"
    except sr.RequestError as e:
        text = f"Sphinx recognition request failed; {e}"
        
    return text

####################################################################

# Define a function to convert video data to audio
def yt_link2text(youtube_url, output_directory="downloaded", name="xyz"):
    # Create a YouTube object
    yt = YouTube(youtube_url)
    
    # Get the video stream with the highest resolution
    video_stream = yt.streams.get_highest_resolution()
    # Fetch the video data using requests
    video_data = requests.get(video_stream.url).content
    # Generate a unique filename for the temporary video file
    temp_file_name = f"{name}" + ".mp4"
    temp_file_path = f"{output_directory}/{temp_file_name}"
    # Write video data to the temporary video file
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(video_data)
    # Load the video using moviepy
    video_clip = VideoFileClip(temp_file_path)
    # Generate a unique filename for the audio file
    audio_file_name = f"{name}" + ".wav"
    audio_file_path = f"{output_directory}/{audio_file_name}"
    # Save the audio as a WAV file in the output directory
    video_clip.audio.write_audiofile(audio_file_path)
    # Close the video clip
    video_clip.close()
    
    text = audio2text(audio_file_path)
    
    return text

####################################################################

def video2text(video_file, output_audio_file = "downloaded/vid2aud.wav"):
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_file, codec='pcm_s16le')
    text = audio2text(output_audio_file)
    
    return text

####################################################################

def main():
    st.title("Multitype Summarization App")

    script_directory = pathlib.Path(__file__).parent

    # Text Section
    st.markdown("---")
    st.markdown('<h2 style="color: yellow;">Text Summarization</h2>', unsafe_allow_html=True)
    st.write("Simplify and summarize! Enter or upload text, and watch as we condense the information, distilling it into a comprehensible and concise format.")
    entered_text = st.text_area("Enter Text")
    if entered_text:
        if st.button("Summarize Text"):
            
            try:
                text = summarize_text(preprocess_text(entered_text))
            except:
                text = "Mistake on our part. While summarizing, we are actively addressing and rectifying the issue."
            
            st.text_area("Text:", text, height=200)

    # Image Section
    st.markdown("---")
    st.markdown('<h2 style="color: yellow;">Image Summarization</h2>', unsafe_allow_html=True)
    st.write(
        "Bring images to life! Upload a picture, and we'll extract the textual information, providing a summarized overview that captures the essence of your visual content. It is the initial phase of this project, It may not give best summary, It's worth noting that text extraction from images requires clear and discernible text. Ensure that the images contain legible text to facilitate effective summarization, much like the requirement for PDFs.")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.write(f"Image File: {uploaded_image.name}")
        if st.button("Summarize Image"):
            image_path = save_uploaded_file(uploaded_image, script_directory, "png", "xyz")
            
            try:
                image_text = summarize_text(img2text(image_path))
            except:
                image_text = "In the initial phase of this project, it's worth noting that text extraction from images requires clear and discernible text. Ensure that the images contain legible text to facilitate effective summarization, much like the requirement for PDFs."
            
            st.text_area("Image Text:", image_text, height=200)

    # PDF Section
    st.markdown("---")
    st.markdown('<h2 style="color: yellow;">PDF Summarization</h2>', unsafe_allow_html=True)
    st.write("Unleash the power of your PDFs! Upload, and witness the magic as we distill the essence, delivering a concise and powerful summary right before your eyes. This project is in its initial stage, It may not give best summary, it's important to highlight that text extraction from PDFs requires clear and visible text. Please ensure that the PDF contains legible text for effective summarization.")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf is not None:
        st.write(f"PDF File: {uploaded_pdf.name}")
        if st.button("Summarize PDF"):
            pdf_path = save_uploaded_file(uploaded_pdf, script_directory, "pdf", "xyz")
            
            try:
                pdf_text = summarize_text(pdf2text(pdf_path))
            except:
                pdf_text = "This project is in its initial stage, it's important to highlight that text extraction from PDFs requires clear and visible text. Please ensure that the PDF contains legible text for effective summarization."
            
            st.text_area("PDF Text:", pdf_text, height=200)

    # Link Section
    st.markdown("---")
    st.markdown('<h2 style="color: yellow;">Youtube Video Summarization</h2>', unsafe_allow_html=True)
    st.write(
        "Unleash the power of video content! Share a YouTube link, and we'll transform the spoken words into a summarized transcript, making the information easily digestible. It is the initial phase of this project, It may not give best summary.")
    entered_link = st.text_input("Enter Link (Text)")
    if entered_link:
        if st.button("Summarize Link"):
            
            try:
                yt_text = summarize_text(yt_link2text(entered_link))
            except:
                yt_text = "Mistake on our part. While summarizing, we are actively addressing and rectifying the issue."
            
            st.text_area("YouTube Video Text:", yt_text, height=200)

    # Video Section
    st.markdown("---")
    st.markdown('<h2 style="color: yellow;">Video Summarization</h2>', unsafe_allow_html=True)
    st.write("Drop a video, and we'll give you the quick rundown. See the highlights, skip the fluff. Simple, straightforward summarization, just a click away.As this project is in its initial stage,It may not give best summary, it's crucial to note that our text extraction relies on clear and noise-free audio. Please ensure that the audio quality is optimal for accurate results.")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4"])
    if uploaded_video is not None:
        st.write(f"Video File: {uploaded_video.name}")
        if st.button("Summarize Video"):
            video_path = save_uploaded_file(uploaded_video, script_directory, "mp4", "xyz")
            
            try:
                video_text = summarize_text(video2text(video_path))
            except:
                video_text = "As this project is in its initial stage, it's crucial to note that our text extraction relies on clear and noise-free audio. Please ensure that the audio quality is optimal for accurate results."
            
            st.text_area("Video Text:", video_text, height=200)

    # Audio Section
    st.markdown("---")
    st.markdown('<h2 style="color: yellow;">Audio Summarization</h2>', unsafe_allow_html=True)
    st.write("Amplify your understanding! Upload an audio file, and let us transcribe and summarize the content, transforming complex audio data into clear and concise insights. As this project is still in its early stages,It may not give best sumary, It's essential to emphasize that text extraction from audio relies on clear and noise-free audio. Ensure that the audio quality is optimal to achieve accurate summarization results.")
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav"])
    if uploaded_audio is not None:
        st.write(f"Audio File: {uploaded_audio.name}")
        if st.button("Summarize Audio"):
            audio_path = save_uploaded_file(uploaded_audio, script_directory, "audio", "xyz")
            
            try:
                audio_text = summarize_text(audio2text(audio_path))
            except:
                audio_text = "As this project is still in its early stages, it's essential to emphasize that text extraction from audio relies on clear and noise-free audio. Ensure that the audio quality is optimal to achieve accurate summarization results."
            
            st.text_area("Audio Text:", audio_text, height=200)

def save_uploaded_file(uploaded_file, script_directory, file_type, folder_name):
    temp_dir = pathlib.Path(script_directory, "downloaded", folder_name)
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = tempfile.NamedTemporaryFile(suffix=f".{file_type}", dir=temp_dir, delete=False, prefix="xyz").name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    return file_path

####################################################################

if __name__ == "__main__":
    main()