import streamlit as st
from transformers import pipeline
import librosa
import re
import time

# Set Streamlit Page Configuration
st.set_page_config(page_title="🔤 Alphabet Adventure for Kids", layout="centered")

# Load speech recognition pipeline using librosa backend (prevents ffmpeg crashes)
@st.cache_resource
def load_speech_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

transcriber = load_speech_model()

# Full 26 Letters Data
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
    {"letter": "Z", "word": "Zebra", "valid": ["z", "zed", "zee", "zebra", "ज़ेड", "ज़ेब्रा"], "image": "https://img.freepik.com/free-vector/alphabet-z-is-zebra_1308-79094.jpg"}
]

SURPRISE_GIFTS = {
    5: "🎁 बहुत बढ़िया! आपको मिला STAR BADGE!",
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
if "is_correct" not in st.session_state:
    st.session_state.is_correct = False
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# Header & Scoreboard
st.title("🔤 बच्चों का बोलना सीखो ऐप")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🔥 Streak", value=st.session_state.streak)
with col2:
    st.metric(label="⭐ Total Score", value=st.session_state.score)
with col3:
    elapsed_time = int(time.time() - st.session_state.start_time)
    st.metric(label="⏱️ Timer", value=f"{elapsed_time}s")

st.markdown("---")

current_item = ALPHABET[st.session_state.idx]

# Flashcard Container
with st.container(border=True):
    st.subheader(f"Card {st.session_state.idx + 1} of {len(ALPHABET)}")
    
    card_col1, card_col2 = st.columns([1, 1])
    with card_col1:
        st.image(current_item["image"], use_container_width=True)
    with card_col2:
        st.markdown(f"# **{current_item['letter']}**")
        st.markdown(f"### for **{current_item['word']}**")
        st.info(f"👉 **Speak:** '{current_item['letter']}' or '{current_item['word']}'")

# Audio Input
audio_file = st.audio_input("🎙️ माइक दबाकर अपनी आवाज रिकॉर्ड करें")

if audio_file is not None and not st.session_state.is_correct:
    # Save recording locally
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    
    # Load audio array via librosa to bypass ffmpeg dependency
    audio_data, sampling_rate = librosa.load("temp_audio.wav", sr=16000)
    
    # Transcribe speech with Hugging Face Whisper pipeline
    result = transcriber(audio_data)
    spoken_text = result["text"].strip().lower()
    spoken_text = re.sub(r'[^\w\s]', '', spoken_text)
    
    st.write(f"🗣️ **You said:** *'{spoken_text}'*")
    
    # Check correctness
    if any(val in spoken_text for val in current_item["valid"]):
        st.session_state.is_correct = True
        st.session_state.streak += 1
        st.session_state.score += 1
        st.success("✅ शाबाश! सही जवाब! 'Next' बटन दबाकर आगे बढ़ें.")
        
        # Surprise Rewards
        if st.session_state.streak in SURPRISE_GIFTS:
            st.balloons()
            st.info(SURPRISE_GIFTS[st.session_state.streak])
    else:
        st.session_state.streak = 0
        st.error(f"❌ फिर से कोशिश करो! बोलो '{current_item['letter']}' या '{current_item['word']}'")

# Progression Control: Next button enables ONLY after correct answer
st.markdown("---")
btn_col1, btn_col2 = st.columns([1, 1])

with btn_col1:
    if st.button("गेम दोबारा शुरू करें (Restart) 🔄"):
        st.session_state.idx = 0
        st.session_state.streak = 0
        st.session_state.score = 0
        st.session_state.is_correct = False
        st.session_state.start_time = time.time()
        st.rerun()

with btn_col2:
    if st.button("अगला अक्षर (Next) ➡️", disabled=not st.session_state.is_correct):
        if st.session_state.idx + 1 < len(ALPHABET):
            st.session_state.idx += 1
            st.session_state.is_correct = False
            st.rerun()
        else:
            st.snow()
            st.title("🏆 GRAND FINALE TROPHY UNLOCKED! YOU WIN! 🏆")
