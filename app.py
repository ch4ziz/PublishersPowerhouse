import streamlit as st
import requests
import pandas as pd # Not strictly needed for this version, but often useful
import openai
from ebooklib import epub
from io import BytesIO # To handle image data in memory
from PIL import Image, ImageDraw, ImageFont # For fallback cover image generation
import logging
import json
import time
from typing import List

# -------------------------------
# Configuration & API Keys Setup
# -------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key from Streamlit secrets
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit Secrets. Please add it to run this app.")
    st.stop() # Stop execution if API key is missing

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="AI-Powered Book Creator", layout="wide")

st.title("✍️ AI-Powered Book Creator")
st.markdown(
    """
    Generate a complete book from a topic, including an AI-generated outline,
    detailed chapters, and a custom cover image, compiled into an EPUB file.

    > **Disclaimer**: AI-generated content may contain inaccuracies. Please review and edit the generated book.
    > Using DALL-E and GPT-4 incurs costs on your OpenAI account.
    """
)

# -------------------------------
# Helper Functions for OpenAI API Calls
# -------------------------------

def call_openai_chat_api(prompt: str, model: str = "gpt-4o", max_tokens: int = 1500, temperature: float = 0.7) -> str:
    """
    Calls the OpenAI ChatCompletion API and returns the response text.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
        )
        result = response.choices[0].message['content'].strip()
        logger.debug("Received response from OpenAI Chat API")
        return result
    except openai.error.AuthenticationError:
        st.error("OpenAI API Key is invalid or expired. Please update it in Streamlit Secrets.")
        st.stop()
    except openai.error.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        st.error(f"OpenAI API Error: {e}. Please try again later.")
        st.stop()
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        st.error(f"An unexpected error occurred with the OpenAI API: {e}")
        st.stop()

def call_openai_image_api(prompt: str, size: str = "1024x1024", quality: str = "standard") -> BytesIO:
    """
    Calls the OpenAI Image (DALL-E) API and returns the image data as BytesIO.
    """
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1
        )
        image_url = response.data[0].url
        logger.info(f"Generated DALL-E image URL: {image_url}")
        
        img_data = requests.get(image_url).content
        return BytesIO(img_data)
    except openai.error.OpenAIError as e:
        logger.error(f"DALL-E API error: {e}")
        st.error(f"Could not generate cover image using DALL-E: {e}. Try adjusting the topic or using a simpler one.")
        raise
    except Exception as e:
        logger.error(f"Error fetching DALL-E image: {e}")
        st.error(f"An unexpected error occurred while fetching the DALL-E image: {e}")
        raise

# -------------------------------
# Book Generation Functions
# -------------------------------

def generate_book_outline(topic: str, num_chapters: int) -> List[str]:
    """
    Generates a structured outline (in JSON) for the book.
    """
    outline_prompt = (
        f"Generate a JSON array with exactly {num_chapters} strings. "
        f"Each string should be a concise and engaging chapter title for a non-fiction book on the topic: '{topic}'. "
        "Do not include any extra text, numbering, or sub-chapter points outside the JSON array. Return only the JSON array."
    )
    outline_response = call_openai_chat_api(outline_prompt, max_tokens=num_chapters * 30, model="gpt-4o") # Adjust max_tokens based on num_chapters
    
    try:
        chapters = json.loads(outline_response)
        if not isinstance(chapters, list):
            raise ValueError("The output is not a JSON array.")
        
        # Robustly handle cases where AI might not return exact number of chapters
        if len(chapters) != num_chapters:
            logger.warning(f"Expected {num_chapters} chapters, but got {len(chapters)} from AI. Adjusting.")
            if len(chapters) > num_chapters:
                chapters = chapters[:num_chapters]
            else:
                while len(chapters) < num_chapters:
                    chapters.append(f"Chapter {len(chapters) + 1}: Continuation of {topic}")
        return chapters
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON outline: {e}. Raw response: {outline_response[:500]}...")
        st.error(f"AI returned an invalid outline. Please try again or refine the topic. Error: {e}")
        st.stop()
    except Exception as e:
        logger.error(f"Error processing outline: {e}")
        st.error(f"An error occurred while processing the book outline: {e}")
        st.stop()


def generate_chapter_content(topic: str, chapter_title: str, chapter_idx: int) -> str:
    """
    Generates detailed narrative content for a single chapter.
    """
    chapter_prompt = (
        f"Write a detailed and engaging narrative chapter for a book on '{topic}'. "
        f"The chapter title is '{chapter_title}'. Focus on providing a thorough discussion "
        "with full paragraphs, rich explanations, and smooth transitions between ideas. "
        "Avoid using bullet points, lists, or fragmented points; focus on creating a flowing narrative "
        "that fully explores the topic. Make it around 1500 words. " # Request more words for better content
        "Do not include the chapter title in the response, only the content."
    )
    chapter_text = call_openai_chat_api(chapter_prompt, max_tokens=4000, model="gpt-4o", temperature=0.7) # Increased max_tokens for longer chapters
    
    # Format chapter content in HTML paragraphs, handling common AI formatting
    chapter_html = f"<h2>Chapter {chapter_idx}: {chapter_title}</h2>\n"
    # Replace common newline patterns with proper HTML paragraphs
    paragraphs = chapter_text.split('\n\n')
    formatted_paragraphs = [f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()]
    chapter_html += "\n".join(formatted_paragraphs)
    
    return chapter_html

def generate_cover_image_data(topic: str) -> BytesIO:
    """
    Generates a book cover image using DALL-E or a PIL fallback.
    Returns image data as BytesIO.
    """
    cover_prompt = (
        f"A visually striking and relevant book cover illustration for a non-fiction book "
        f"titled 'A Comprehensive Guide to {topic}'. The style should be modern and inviting. "
        "Focus on abstract concepts or symbolic representations related to the topic, "
        "avoiding any text on the image. High detail, vibrant colors."
    )
    
    try:
        with st.spinner("Generating AI-powered book cover..."):
            image_data = call_openai_image_api(cover_prompt, size="1024x1024", quality="hd")
            st.success("Cover image generated by DALL-E!")
            return image_data
    except Exception as e:
        logger.error(f"Failed to generate DALL-E cover: {e}. Falling back to placeholder.")
        st.warning("Failed to generate AI cover image. Creating a simple placeholder cover.")
        # Fallback to PIL-generated placeholder
        img_size = (600, 800)
        img = Image.new('RGB', img_size, color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        
        try:
            # Try to use a common font or default to a generic one
            font_path = "arial.ttf" # Common font on Windows
            if sys.platform == "darwin": # Mac
                font_path = "/Library/Fonts/Arial.ttf"
            # Fallback for Linux or if Arial not found
            font = ImageFont.truetype(font_path, 40)
        except IOError:
            font = ImageFont.load_default() # Load default PIL font

        text = f"Book Cover\n'{topic}'"
        # Calculate text size and position
        text_bbox = d.textbbox((0,0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (img_size[0] - text_width) / 2
        y = (img_size[1] - text_height) / 2
        
        d.multiline_text((x, y), text, fill=(255,255,255), font=font, align="center")
        
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        byte_arr.seek(0)
        return byte_arr


def compile_book_to_epub(title: str, author: str, content: str, cover_image_data: BytesIO, output_file_name: str) -> BytesIO:
    """
    Compiles the book's content and cover image data into an EPUB file in memory.
    Returns the EPUB file data as BytesIO.
    """
    logger.info("Compiling content into EPUB format...")
    try:
        book = epub.EpubBook()
        book.set_identifier(f"book_{datetime.now().timestamp()}") # Unique ID
        book.set_title(title)
        book.set_language('en')
        book.add_author(author)
        
        # Add a cover image item
        cover_image_filename = "cover.png" # Standard filename for EPUB cover
        book.add_item(epub.EpubItem(uid="cover_image", file_name=cover_image_filename, media_type="image/png", content=cover_image_data.read()))
        book.set_cover(cover_image_filename, cover_image_data.read()) # Set cover in EPUB metadata

        # Create the main content chapter
        main_chapter = epub.EpubHtml(title="Main Content", file_name="main_content.xhtml", lang='en')
        main_chapter.content = f"<h1>{title}</h1>\n" + content
        book.add_item(main_chapter)
        
        # Define table of contents and spine (reading order)
        book.toc = (epub.Link('main_content.xhtml', 'Start', 'main_content'),)
        book.add_item(epub.EpubNcx()) # Navigational control file
        book.add_item(epub.EpubNav()) # Navigation document
        
        # Define the spine (order of content in the book)
        book.spine = ['cover', 'nav', main_chapter]
        
        # Write EPUB to BytesIO
        epub_buffer = BytesIO()
        epub.write_epub(epub_buffer, book, {}) # Pass empty dict for options
        epub_buffer.seek(0) # Rewind to the beginning of the buffer
        
        logger.info(f"EPUB file created in memory: {output_file_name}")
        return epub_buffer
    except Exception as e:
        logger.error(f"Failed to compile EPUB: {e}")
        st.error(f"Failed to compile EPUB file: {e}")
        st.stop()


# -------------------------------
# Streamlit UI & Main Logic
# -------------------------------

st.sidebar.header("Book Details")
topic = st.sidebar.text_input("Book Topic:", "The History of AI").strip()
title = st.sidebar.text_input("Book Title:", "AI: From Concept to Reality").strip()
author = st.sidebar.text_input("Author Name:", "AI Writer").strip()
num_chapters = st.sidebar.number_input("Number of Chapters:", min_value=3, max_value=10, value=5, step=1)
output_filename = st.sidebar.text_input("Output EPUB Filename:", "my_ai_book.epub").strip()
if not output_filename.endswith(".epub"):
    output_filename += ".epub"

st.sidebar.markdown("---")
if st.sidebar.button("✨ Create Book"):
    if not topic or not title or not author:
        st.error("Please fill in all book details (Topic, Title, Author).")
    else:
        st.subheader(f"Creating Your Book: '{title}' by {author}")
        
        with st.status("Starting book creation...", expanded=True) as status_box:
            st.write("Generating book outline...")
            outline = generate_book_outline(topic, num_chapters)
            st.json({"Book Outline": outline}) # Display outline in JSON for clarity
            st.info(f"Generated outline with {len(outline)} chapters.")

            st.write("Generating chapter content...")
            full_book_content = ""
            chapter_placeholders = [st.empty() for _ in outline] # Placeholders for chapter status
            for idx, chapter_title in enumerate(outline):
                chapter_placeholders[idx].info(f"Generating Chapter {idx+1}: {chapter_title}...")
                chapter_html = generate_chapter_content(topic, chapter_title, idx + 1)
                full_book_content += chapter_html + "\n\n"
                chapter_placeholders[idx].success(f"Chapter {idx+1}: '{chapter_title}' generated.")
                time.sleep(1) # Small pause to be gentle on API limits and show progress

            st.write("Generating book cover...")
            cover_image_buffer = generate_cover_image_data(topic)
            st.image(cover_image_buffer.getvalue(), caption="Generated Book Cover Preview", use_column_width=True) # Display generated image
            cover_image_buffer.seek(0) # Reset buffer for EPUB compilation

            st.write("Compiling EPUB file...")
            epub_file_buffer = compile_book_to_epub(title, author, full_book_content, cover_image_buffer, output_filename)
            status_box.update(label="Book creation complete!", state="complete", expanded=False)

            st.success("Your book has been created!")
            st.download_button(
                label=f"📥 Download '{output_filename}'",
                data=epub_file_buffer.getvalue(),
                file_name=output_filename,
                mime="application/epub+zip",
                key="download_epub_button"
            )
            epub_file_buffer.close() # Close the BytesIO buffer
            cover_image_buffer.close() # Close the BytesIO buffer
            st.balloons() # Celebrate!

st.markdown("---")
st.markdown("### How to Use:")
st.markdown(
    """
    1.  Enter the main topic for your book.
    2.  Provide a suitable title and author name.
    3.  Choose the desired number of chapters (between 3 and 10).
    4.  Specify a filename for your EPUB output.
    5.  Click "Create Book" and wait for the AI to do its magic!
    6.  Download the generated EPUB file.
    """
)
