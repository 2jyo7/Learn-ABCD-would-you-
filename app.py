import streamlit as st
from transformers import pipeline
from gtts import gTTS
import re
import os

# Set Streamlit Page Configuration
st.set_page_config(page_title="🔤 Alphabet Adventure for Kids", layout="centered")

# Load lightweight speech recognition pipeline
@st.cache_resource
def load_speech_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

transcriber = load_speech_model()

# Full 26 Letters List with Hindi targets and images
ALPHABET = [
    {"letter": "A", "word": "Apple", "valid": ["a", "apple", "ए", "एप्पल"], "image": "https://img.freepik.com/free-vector/alphabet-a-is-apple_1308-78229.jpg"},
    {"letter": "B", "word": "Ball", "valid": ["b", "bee", "ball", "बी", "बॉल"], "image": "https://img.freepik.com/free-vector/alphabet-b-is-ball_1308-78709.jpg"},
    {"letter": "C", "word": "Cat", "valid": ["c", "see", "cat", "सी", "कैट"], "image": "https://img.freepik.com/free-vector/alphabet-c-is-cat_1308-78722.jpg"},
    {"letter": "D", "word": "Dog", "valid": ["d", "dee", "dog", "डी", "डॉग"], "image": "https://img.freepik.com/free-vector/alphabet-d-is-dog_1308-78801.jpg"},
    {"letter": "E", "word": "Elephant", "valid": ["e", "elephant", "ई", "एलिफेंट"], "image": "https://img.freepik.com/free-vector/alphabet-e-is-elephant_1308-78810.jpg"},
    {"letter": "F", "word": "Fish", "valid": ["f", "eff", "fish", "एफ", "फिश"], "image": "https://img.freepik.com/free-vector/alphabet-f-is-fish_1308-78835.jpg"},
    {"letter": "G", "word": "Grapes", "valid": ["g", "gee", "grapes", "जी", "ग्रेप्स"], "image": "https://img.freepik.com/free-vector/alphabet-g-is-grapes_1308-78848.jpg"},
    {"letter": "H", "word": "Hat", "valid": ["h", "aitch", "hat", "एच", "हैट"], "image": "https://img.freepik.com/free-vector/alphabet-h-is-hat_1308-78861.jpg"},
    {"letter": "I", "word": "Ice cream", "valid": ["i", "eye", "ice cream", "आई", "आइसक्रीम"], "image": "https://img.freepik.com/free-vector/alphabet-i-is-ice-cream_1308-78873.jpg"},
    {"letter": "J", "word": "Juice", "valid": ["j", "jay", "juice", "जे", "जूस"], "image": "https://img.freepik.com/free-vector/alphabet-j-is-juice_1308-78886.jpg"},
    {"letter": "K", "word": "Kite", "valid": ["k", "kay", "kite", "के", "काइट"], "image": "https://img.freepik.com/free-vector/alphabet-k-is-kite_1308-78899.jpg"},
    {"letter": "L", "word": "Lion", "valid": ["l", "el", "lion", "एल", "लायन"], "image": "https://img.freepik.com/free-vector/alphabet-l-is-lion_1308-78912.jpg"},
    {"letter": "M", "word": "Monkey", "valid": ["m", "em", "monkey", "एम", "मंकी"], "image": "https://img.freepik.com/free-vector/alphabet-m-is-monkey_1308-78925.jpg"},
    {"letter": "N", "word": "Nest", "valid": ["n", "en", "nest", "एन", "नेस्ट"], "image": "https://img.freepik.com/free-vector/alphabet-n-is-nest_1308-78938.jpg"},
    {"letter": "O", "word": "Orange", "valid": ["o", "oh", "orange", "ओ", "ऑरेंज"], "image": "https://img.freepik.com/free-vector/alphabet-o-is-orange_1308-78951.jpg"},
    {"letter": "P", "word": "Parrot", "valid": ["p", "pee", "parrot", "पी", "पैरेट"], "image": "https://img.freepik.com/free-vector/alphabet-p-is-parrot_1308-78964.jpg"},
    {"letter": "Q", "word": "Queen", "valid": ["q", "cue", "queen", "क्यू", "क्वीन"], "image": "https://img.freepik.com/free-vector/alphabet-q-is-queen_1308-78977.jpg"},
    {"letter": "R", "word": "Rabbit", "valid": ["r", "ar", "rabbit", "आर", "रैबिट"], "image": "https://img.freepik.com/free-vector/alphabet-r-is-rabbit_1308-78990.jpg"},
    {"letter": "S", "word": "Sun", "valid": ["s", "ess", "sun", "एस", "सन"], "image": "https://img.freepik.com/free-vector/alphabet-s-is-sun_1308-79003.jpg"},
    {"letter": "T", "word": "Tiger", "valid": ["t", "tee", "tiger", "टी", "टाइगर"], "image": "https://img.freepik.com/free-vector/alphabet-t-is-tiger_1308-79016.jpg"},
    {"letter": "U", "word": "Umbrella", "valid": ["u", "you", "umbrella", "यू", "अम्ब्रेला"], "image": "https://img.freepik.com/free-vector/alphabet-u-is-umbrella_1308-79029.jpg"},
    {"letter": "V", "word": "Van", "valid": ["v", "vee", "van", "वी", "वैन"], "image": "https://img.freepik.com/free-vector/alphabet-v-is-van_1308-79042.jpg"},
    {"letter": "W", "word": "Watch", "valid": ["w", "double u", "watch", "डबल यू", "वॉच"], "image": "https://img.freepik.com/free-vector/alphabet-w-is-watch_1308-79055.jpg"},
    {"letter": "X", "word": "Xylophone", "valid": ["x", "ex", "xylophone", "एक्स", "जाइलोफोन"], "image": "https://img.freepik.com/free-vector/alphabet-x-is-xylophone_1308-79068.jpg"},
    {"letter": "Y", "word": "Yak", "valid": ["y", "why", "yak", "वाई", "याक"], "image": "https://img.freepik.com/free-vector/alphabet-y-is-yak_1308-79081.jpg"},
    {"letter": "Z", "word": "Zebra", "valid": ["z", "zed", "zee", "zebra", "ज़ेड", "ज़ेब्रा"], "image": "https://img.freepik.com/free-vector/alphabet-z-is-zebra_1308-79094.jpg"}
]

SURPRISE_GIFTS = {
    5: "🎁 बहुत बढ़िया! आपको मिला STAR BADGE!",
    10: "🎈 वाह! आपको मिले BALLOONS!",
    15: "🎨 शानदार! आपको मिली VIRTUAL CRAYONS!",
    20: "👑 कमाल कर दिया! आपको मिला CROWN!",
    25: "🚀 स्पेस रॉकेट बैज मिला!"
}

# Session state initialization
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "score" not in st.session_state:
    st.session_state.score = 0

def speak_hindi(text):
    tts = gTTS(text=text, lang="hi")
    tts.save("response.mp3")
    st.audio("response.mp3", autoplay=True)

st.title("🔤 बच्चों का बोलना सीखो ऐप")
st.markdown("---")

current_item = ALPHABET[st.session_state.idx]

# Display Card Image
st.image(current_item["image"], caption=f"Letter {current_item['letter']} for {current_item['word']}", use_container_width=True)

# Scores UI
col1, col2 = st.columns(2)
with col1:
    st.metric(label="🔥 लगातार सही जवाब (Streak)", value=st.session_state.streak)
with col2:
    st.metric(label="⭐ कुल स्कोर (Total Score)", value=st.session_state.score)

# Native Microphone Audio Input
audio_file = st.audio_input("🎙️ माइक दबाकर अपनी आवाज रिकॉर्ड करें")

if audio_file is not None:
    # Save recording to temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    
    # Transcribe speech
    result = transcriber("temp_audio.wav")
    spoken_text = result["text"].strip().lower()
    spoken_text = re.sub(r'[^\w\s]', '', spoken_text)
    
    is_correct = any(val in spoken_text for val in current_item["valid"])
    
    if is_correct:
        st.session_state.streak += 1
        st.session_state.score += 1
        st.success(f"✅ शाबाश! सही जवाब!")
        
        # Check for streak gifts
        if st.session_state.streak in SURPRISE_GIFTS:
            gift = SURPRISE_GIFTS[st.session_state.streak]
            st.balloons()
            st.info(gift)
            speak_hindi(f"शाबाश! {gift}")
        else:
            speak_hindi("शाबाश! आपका जवाब सही है!")
            
        # Check Grand Finale completion
        if st.session_state.idx + 1 >= len(ALPHABET):
            st.snow()
            st.title("🏆 GRAND FINALE TROPHY UNLOCKED! YOU WIN! 🏆")
            speak_hindi("बधाई हो! आपने सारे अक्षर सीख लिए हैं! आपको मिलता है ग्रैंड ट्रॉफी!")
        else:
            st.session_state.idx += 1
            st.rerun()
    else:
        st.session_state.streak = 0
        st.error(f"❌ फिर से कोशिश करो! बोलो '{current_item['letter']}' या '{current_item['word']}'")
        speak_hindi(f"फिर से कोशिश करो! बोलो {current_item['letter']} फॉर {current_item['word']}")

if st.button("गेम दोबारा शुरू करें (Restart) 🔄"):
    st.session_state.idx = 0
    st.session_state.streak = 0
    st.session_state.score = 0
    st.rerun()
