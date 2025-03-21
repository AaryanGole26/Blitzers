import streamlit as st
import os
from gtts import gTTS
import speech_recognition as sr
import tempfile
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# -------------------- Text-to-Speech Function --------------------
def text_to_speech(text, language='en'):
    """
    Convert text to speech and play it.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_audio_file.name)
            st.audio(temp_audio_file.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")

# -------------------- Speech-to-Text Function --------------------
def speech_to_text():
    """
    Convert speech to text using the microphone.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds
            try:
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                st.warning("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {str(e)}")
    except Exception as e:
        st.error(f"Error in speech-to-text conversion: {str(e)}")
    return None

# System prompt for the financial analyst
SYSTEM_PROMPT = (
    """You are an AI financial analyst. Maintain context and provide insightful responses. "
    "Do not display the thinking process, internal thoughts, or reasoning. "
    "Only return the final response directly to the user."
    You are an advanced AI financial analyst. Your task is to generate in-depth financial insights, trend analysis, and investment strategies based on user inputs. Follow the structured format below and provide actionable recommendations:

1. **Market Trend Analysis**  
   - Analyze market trends based on historical data, economic indicators, and global events.  
   - Identify emerging opportunities and potential risks.  
   - Provide key insights, trend forecasts, and strategic implications.

2. **Company-Specific Financial Projection**  
   - Assess a company's past financial performance and project future growth.  
   - Include revenue forecasts, profitability analysis, and risk assessments.  
   - Offer strategic recommendations based on industry trends.

3. **Investment Strategy Recommendations**  
   - Suggest an optimal investment strategy based on current market conditions and investor risk profile.  
   - Provide asset allocation strategies and diversification recommendations.  
   - Highlight key risks and how to mitigate them.

4. **Comprehensive Financial Analysis Report**  
   - Generate a detailed report based on provided financial data, market conditions, and strategic goals.  
   - Include key insights, risk assessment, and actionable recommendations.  
   - Format the report in structured sections (Executive Summary, Market Analysis, Company Performance, Risk Assessment, Investment Strategy, Conclusion).

Be concise, data-driven, and ensure responses are structured for clarity and decision-making.
""")
# Initialize Ollama model
model = OllamaLLM(model="deepseek-r1:1.5b")

def get_ai_response(prompt):
    """
    Get response from Ollama model
    """
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Query: {prompt}"
        # The invoke() method returns a string directly, but we need to ensure it's properly formatted
        response = model.invoke(full_prompt)
        
        # Handle potential empty or invalid responses
        if not response or not isinstance(response, str):
            return "I apologize, but I couldn't generate a proper response at this time."
            
        # Clean up the response if needed (remove any extra whitespace or formatting)
        response = response.strip()
        
        return response
    except Exception as e:
        st.error(f"Error details: {str(e)}")
        return "I apologize, but there was an error generating the response. Please try again."

def main():
    st.set_page_config(page_title="AI Financial Analyst", page_icon="💹")
    
    # Apply custom styling
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton button {
            background-color: #2E8B57;
            color: white;
            border-radius: 5px;
        }
        .chat-container {
            padding: 20px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("💹 AI Financial Analyst")
    st.markdown("Your intelligent companion for financial analysis and investment strategies.")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize session state for user input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with speech-to-text option
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Use the session state value as the default value for the text input
        user_input = st.text_input("Ask your financial question...", 
                                 value=st.session_state.user_input,
                                 key="text_input_widget")
    
    with col2:
        if st.button("🎤", help="Click to speak your question"):
            with st.spinner("Listening..."):
                spoken_text = speech_to_text()
                if spoken_text:
                    # Update the session state
                    st.session_state.user_input = spoken_text
                    # Rerun the app to show the new input
                    st.rerun()

    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("Analyzing..."):
            ai_response = get_ai_response(user_input)
            
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            # Display the latest response
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                
                # Add text-to-speech option for the response
                if st.button("🔊 Listen to Response", key=f"tts_{len(st.session_state.messages)}"):
                    text_to_speech(ai_response)
        
        # Clear the input after processing
        st.session_state.user_input = ""

    # Clear chat button
    if st.button("Generate Chat"):
        st.session_state.messages = []
        st.session_state.user_input = ""
        st.rerun()

if __name__ == "__main__":
    main()
