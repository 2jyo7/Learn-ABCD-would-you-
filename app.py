import streamlit as st
from transformers import pipeline
import librosa
import re
import time

# Set Streamlit Page Configuration
st.set_page_config(page_title="🔤 Alphabet Adventure for Kids", layout="centered")

# Load speech recognition pipeline using librosa backend
@st.cache_resource
def load_speech_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

transcriber = load_speech_model()

# Full 26 Letters Data mapped with relevant image URLs
ALPHABET = [
    {"letter": "A", "word": "Apple", "valid": ["a", "apple", "ए", "एप्पल"], "image": "https://img.freepik.com/free-vector/isolated-delicious-apple-cartoon_1308-133602.jpg"},
    {"letter": "B", "word": "Ball", "valid": ["b", "bee", "ball", "बी", "बॉल"], "image": "https://img.freepik.com/free-vector/colorful-ball-cartoon-style_1308-133202.jpg"},
    {"letter": "C", "word": "Cat", "valid": ["c", "see", "cat", "सी", "कैट"], "image": "https://img.freepik.com/free-vector/cute-cat-sitting-cartoon-vector-icon-illustration_138676-2313.jpg"},
    {"letter": "D", "word": "Dog", "valid": ["d", "dee", "dog", "डी", "डॉग"], "image": "https://img.freepik.com/free-vector/cute-dog-sitting-cartoon-vector-icon-illustration_138676-2312.jpg"},
    {"letter": "E", "word": "Elephant", "valid": ["e", "elephant", "ई", "एलिफेंट"], "image": "https://img.freepik.com/free-vector/cute-elephant-sitting-cartoon-vector-icon-illustration_138676-2220.jpg"},
    {"letter": "F", "word": "Fish", "valid": ["f", "eff", "fish", "एफ", "फिश"], "image": "https://img.freepik.com/free-vector/cute-fish-swimming-cartoon-vector-icon-illustration_138676-2216.jpg"},
    {"letter": "G", "word": "Grapes", "valid": ["g", "gee", "grapes", "जी", "ग्रेप्स"], "image": "https://img.freepik.com/free-vector/fresh-grapes-bunch-cartoon-icon-illustration_138676-2882.jpg"},
    {"letter": "H", "word": "Hat", "valid": ["h", "aitch", "hat", "एच", "हैट"], "image": "https://img.freepik.com/free-vector/stylish-hat-cartoon-vector-icon-illustration_138676-3215.jpg"},
    {"letter": "I", "word": "Ice cream", "valid": ["i", "eye", "ice cream", "आई", "आइसक्रीम"], "image": "https://img.freepik.com/free-vector/delicious-ice-cream-cone-cartoon-vector-icon-illustration_138676-2287.jpg"},
    {"letter": "J", "word": "Juice", "valid": ["j", "jay", "juice", "जे", "जूस"], "image": "https://img.freepik.com/free-vector/orange-juice-glass-cartoon-vector-icon-illustration_138676-2283.jpg"},
    {"letter": "K", "word": "Kite", "valid": ["k", "kay", "kite", "के", "काइट"], "image": "https://img.freepik.com/free-vector/colorful-kite-flying-cartoon-vector-icon-illustration_138676-3190.jpg"},
    {"letter": "L", "word": "Lion", "valid": ["l", "el", "lion", "एल", "लायन"], "image": "https://img.freepik.com/free-vector/cute-lion-sitting-cartoon-vector-icon-illustration_138676-2211.jpg"},
    {"letter": "M", "word": "Monkey", "valid": ["m", "em", "monkey", "एम", "मंकी"], "image": "https://img.freepik.com/free-vector/cute-monkey-sitting-cartoon-vector-icon-illustration_138676-2208.jpg"},
    {"letter": "N", "word": "Nest", "valid": ["n", "en", "nest", "एन", "नेस्ट"], "image": "https://img.freepik.com/free-vector/bird-nest-with-eggs-cartoon-vector-icon-illustration_138676-3180.jpg"},
    {"letter": "O", "word": "Orange", "valid": ["o", "oh", "orange", "ओ", "ऑरेंज"], "image": "https://img.freepik.com/free-vector/fresh-orange-fruit-cartoon-vector-icon-illustration_138676-2879.jpg"},
    {"letter": "P", "word": "Parrot", "valid": ["p", "pee", "parrot", "पी", "पैरेट"], "image": "https://img.freepik.com/free-vector/cute-parrot-sitting-cartoon-vector-icon-illustration_138676-2201.jpg"},
    {"letter": "Q", "word": "Queen", "valid": ["q", "cue", "queen", "क्यू", "क्वीन"], "image": "https://img.freepik.com/free-vector/cute-queen-wearing-crown-cartoon-vector-icon-illustration_138676-3310.jpg"},
    {"letter": "R", "word": "Rabbit", "valid": ["r", "ar", "rabbit", "आर", "रैबिट"], "image": "https://img.freepik.com/free-vector/cute-rabbit-sitting-cartoon-vector-icon-illustration_138676-2189.jpg"},
    {"letter": "S", "word": "Sun", "valid": ["s", "ess", "sun", "एस", "सन"], "image": "https://img.freepik.com/free-vector/cute-sun-smiling-cartoon-vector-icon-illustration_138676-2180.jpg"},
    {"letter": "T", "word": "Tiger", "valid": ["t", "tee", "tiger", "टी", "टाइगर"], "image": "https://img.freepik.com/free-vector/cute-tiger-sitting-cartoon-vector-icon-illustration_138676-2175.jpg"},
    {"letter": "U", "word": "Umbrella", "valid": ["u", "you", "umbrella", "यू", "अम्ब्रेला"], "image": "https://img.freepik.com/free-vector/opened-umbrella-cartoon-vector-icon-illustration_138676-3150.jpg"},
    {"letter": "V", "word": "Van", "valid": ["v", "vee", "van", "वी", "वैन"], "image": "https://img.freepik.com/free-vector/delivery-van-cartoon-vector-icon-illustration_138676-3140.jpg"},
    {"letter": "W", "word": "Watch", "valid": ["w", "double u", "watch", "डबल यू", "वॉच"], "image": "https://img.freepik.com/free-vector/wrist-watch-cartoon-vector-icon-illustration_138676-3130.jpg"},
    {"letter": "X", "word": "Xylophone", "valid": ["x", "ex", "xylophone", "एक्स", "जाइलोफोन"], "image": "https://img.freepik.com/free-vector/colorful-xylophone-cartoon-vector-icon-illustration_138676-3120.jpg"},
    {"letter": "Y", "word": "Yak", "valid": ["y", "why", "yak", "वाई", "याक"], "image": "https://img.freepik.com/free-vector/cute-yak-standing-cartoon-vector-icon-illustration_138676-2150.jpg"},
    {"letter": "Z", "word": "Zebra", "valid": ["z", "zed", "zee", "zebra", "ज़ेड", "ज़ेब्रा"], "image": "https://img.freepik.com/free-vector/cute-zebra-standing-cartoon-vector-icon-illustration_138676-2140.jpg"}
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
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0

# Header & Scoreboard
st.title("🔤 ABCD ऐप")

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

# Audio input (Dynamic key prevents cached audio widget persistence)
audio_file = st.audio_input(
    "🎙️ माइक दबाकर अपनी आवाज रिकॉर्ड करें", 
    key=f"mic_recorder_{st.session_state.idx}_{st.session_state.reset_count}"
)

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    
    # Load audio array via librosa
    audio_data, sampling_rate = librosa.load("temp_audio.wav", sr=16000)
    
    # Transcribe speech
    result = transcriber(audio_data)
    spoken_text = result["text"].strip().lower()
    spoken_text = re.sub(r'[^\w\s]', '', spoken_text)
    
    st.write(f"🗣️ **You said:** *'{spoken_text}'*")
    
    # Check correctness
    if any(val in spoken_text for val in current_item["valid"]):
        st.session_state.is_correct = True
        st.session_state.streak += 1
        st.session_state.score += 1
        st.success("✅ शाबाश! सही जवाब!")
        
        # Surprise Rewards
        if st.session_state.streak in SURPRISE_GIFTS:
            st.balloons()
            st.info(SURPRISE_GIFTS[st.session_state.streak])
    else:
        st.session_state.streak = 0
        st.error(f"❌ फिर से कोशिश करो! बोलो '{current_item['letter']}' या '{current_item['word']}'")

# Navigation Controls
st.markdown("---")
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

with btn_col1:
    if st.button("⬅️ पिछला (Previous)", disabled=st.session_state.idx == 0):
        st.session_state.idx -= 1
        st.session_state.is_correct = True
        st.rerun()

with btn_col2:
    if st.button("गेम दोबारा शुरू करें (Restart) 🔄"):
        st.session_state.idx = 0
        st.session_state.streak = 0
        st.session_state.score = 0
        st.session_state.is_correct = False
        st.session_state.start_time = time.time()
        # Incrementing reset_count gives st.audio_input a brand new key, purging old audio
        st.session_state.reset_count += 1
        st.rerun()

with btn_col3:
    if st.button("अगला (Next) ➡️", disabled=not st.session_state.is_correct):
        if st.session_state.idx + 1 < len(ALPHABET):
            st.session_state.idx += 1
            st.session_state.is_correct = False
            st.rerun()
        else:
            st.snow()
            st.title("🏆 GRAND FINALE TROPHY UNLOCKED! YOU WIN! 🏆")
