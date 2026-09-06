import re
import time
import jellyfish
import librosa
import streamlit as st
from transformers import pipeline

# Set Streamlit Page Configuration
st.set_page_config(page_title="🔤 Alphabet Adventure for Kids", layout="centered")

# Load speech recognition pipeline
@st.cache_resource
def load_speech_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

transcriber = load_speech_model()

ALPHABET = [
    {"letter": "A", "word": "Apple", "image": "https://img.freepik.com/free-vector/isolated-delicious-apple-cartoon_1308-133602.jpg"},
    {"letter": "B", "word": "Ball", "image": "https://img.freepik.com/free-vector/colorful-ball-cartoon-style_1308-133202.jpg"},
    {"letter": "C", "word": "Cat", "image": "https://img.freepik.com/free-vector/cute-cat-sitting-cartoon-vector-icon-illustration_138676-2313.jpg"},
    {"letter": "D", "word": "Dog", "image": "https://img.freepik.com/free-vector/cute-dog-sitting-cartoon-vector-icon-illustration_138676-2312.jpg"},
    {"letter": "E", "word": "Elephant", "image": "https://img.freepik.com/free-vector/cute-elephant-sitting-cartoon-vector-icon-illustration_138676-2220.jpg"},
    {"letter": "F", "word": "Fish", "image": "https://img.freepik.com/free-vector/cute-fish-swimming-cartoon-vector-icon-illustration_138676-2216.jpg"},
    {"letter": "G", "word": "Grapes", "image": "https://img.freepik.com/free-vector/fresh-grapes-bunch-cartoon-icon-illustration_138676-2882.jpg"},
    {"letter": "H", "word": "Hat", "image": "https://img.freepik.com/free-vector/stylish-hat-cartoon-vector-icon-illustration_138676-3215.jpg"},
    {"letter": "I", "word": "Ice cream", "image": "https://img.freepik.com/free-vector/delicious-ice-cream-cone-cartoon-vector-icon-illustration_138676-2287.jpg"},
    {"letter": "J", "word": "Juice", "image": "https://img.freepik.com/free-vector/orange-juice-glass-cartoon-vector-icon-illustration_138676-2283.jpg"},
    {"letter": "K", "word": "Kite", "image": "https://img.freepik.com/free-vector/colorful-kite-flying-cartoon-vector-icon-illustration_138676-3190.jpg"},
    {"letter": "L", "word": "Lion", "image": "https://img.freepik.com/free-vector/cute-lion-sitting-cartoon-vector-icon-illustration_138676-2211.jpg"},
    {"letter": "M", "word": "Monkey", "image": "https://img.freepik.com/free-vector/cute-monkey-sitting-cartoon-vector-icon-illustration_138676-2208.jpg"},
    {"letter": "N", "word": "Nest", "image": "https://img.freepik.com/free-vector/bird-nest-with-eggs-cartoon-vector-icon-illustration_138676-3180.jpg"},
    {"letter": "O", "word": "Orange", "image": "https://img.freepik.com/free-vector/fresh-orange-fruit-cartoon-vector-icon-illustration_138676-2882.jpg"},
    {"letter": "P", "word": "Parrot", "image": "https://img.freepik.com/free-vector/cute-parrot-sitting-cartoon-vector-icon-illustration_138676-2201.jpg"},
    {"letter": "Q", "word": "Queen", "image": "https://img.freepik.com/free-vector/cute-queen-wearing-crown-cartoon-vector-icon-illustration_138676-3310.jpg"},
    {"letter": "R", "word": "Rabbit", "image": "https://img.freepik.com/free-vector/cute-rabbit-sitting-cartoon-vector-icon-illustration_138676-2189.jpg"},
    {"letter": "S", "word": "Sun", "image": "https://img.freepik.com/free-vector/cute-sun-smiling-cartoon-vector-icon-illustration_138676-2180.jpg"},
    {"letter": "T", "word": "Tiger", "image": "https://img.freepik.com/free-vector/cute-tiger-sitting-cartoon-vector-icon-illustration_138676-2175.jpg"},
    {"letter": "U", "word": "Umbrella", "image": "https://img.freepik.com/free-vector/opened-umbrella-cartoon-vector-icon-illustration_138676-3150.jpg"},
    {"letter": "V", "word": "Van", "image": "https://img.freepik.com/free-vector/delivery-van-cartoon-vector-icon-illustration_138676-3140.jpg"},
    {"letter": "W", "word": "Watch", "image": "https://img.freepik.com/free-vector/wrist-watch-cartoon-vector-icon-illustration_138676-3130.jpg"},
    {"letter": "X", "word": "Xylophone", "image": "https://img.freepik.com/free-vector/colorful-xylophone-cartoon-vector-icon-illustration_138676-3120.jpg"},
    {"letter": "Y", "word": "Yak", "image": "https://img.freepik.com/free-vector/cute-yak-standing-cartoon-vector-icon-illustration_138676-2150.jpg"},
    {"letter": "Z", "word": "Zebra", "image": "https://img.freepik.com/free-vector/cute-zebra-standing-cartoon-vector-icon-illustration_138676-2140.jpg"}
]

LETTER_HOMOPHONES = {
    "A": ["a", "eh", "ay", "hey", "hay", "eight", "ei"],
    "B": ["b", "be", "bee"],
    "C": ["c", "see", "sea"],
    "D": ["d", "dee"],
    "E": ["e", "ee"],
    "I": ["i", "eye", "aye"],
    "O": ["o", "oh", "owe"],
    "P": ["p", "pea", "pee"],
    "Q": ["q", "cue", "queue"],
    "R": ["r", "are", "our"],
    "S": ["s", "es"],
    "T": ["t", "tea", "tee"],
    "U": ["u", "you"],
    "Y": ["y", "why"]
}

def validate_pronunciation(spoken_text: str, target_letter: str, target_word: str) -> bool:
    tokens = spoken_text.lower().strip().split()
    if not tokens:
        return False

    target_letter_lower = target_letter.lower()
    target_word_lower = target_word.lower()

    for token in tokens:
        if token == target_letter_lower or token == target_word_lower:
            return True
        if target_letter in LETTER_HOMOPHONES and token in LETTER_HOMOPHONES[target_letter]:
            return True
        if jellyfish.jaro_winkler_similarity(token, target_word_lower) >= 0.82:
            return True
        if jellyfish.metaphone(token) == jellyfish.metaphone(target_word_lower):
            return True

    return False

# Session State Initialization
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
if "attempt_id" not in st.session_state:
    st.session_state.attempt_id = 0
if "error_msg" not in st.session_state:
    st.session_state.error_msg = ""

# Header
st.title("🔤 बच्चों का बोलना सीखो ऐप")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🔥 Streak", value=st.session_state.streak)
with col2:
    st.metric(label="⭐ Total Score", value=st.session_state.score)
with col3:
    st.metric(label="⏱️ Timer", value=f"{int(time.time() - st.session_state.start_time)}s")

st.markdown("---")

current_item = ALPHABET[st.session_state.idx]

with st.container(border=True):
    st.subheader(f"Card {st.session_state.idx + 1} of {len(ALPHABET)}")
    card_col1, card_col2 = st.columns([1, 1])
    with card_col1:
        st.image(current_item["image"], use_container_width=True)
    with card_col2:
        st.markdown(f"# **{current_item['letter']}**")
        st.markdown(f"### for **{current_item['word']}**")
        st.info(f"👉 **Speak:** '{current_item['letter']}' or '{current_item['word']}'")

# Display persisted error message from previous attempt if present
if st.session_state.error_msg:
    st.error(st.session_state.error_msg)

# Dynamic key incorporates card index, reset count, and attempt ID
audio_key = f"mic_{st.session_state.idx}_{st.session_state.reset_count}_{st.session_state.attempt_id}"
audio_file = st.audio_input("🎙️ माइक दबाकर अपनी आवाज रिकॉर्ड करें", key=audio_key)

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    
    audio_data, _ = librosa.load("temp_audio.wav", sr=16000)
    result = transcriber(audio_data)
    spoken_text = re.sub(r'[^\w\s]', '', result["text"].strip().lower())
    
    st.write(f"🗣️ **You said:** *'{spoken_text}'*")
    
    if validate_pronunciation(spoken_text, current_item["letter"], current_item["word"]):
        st.session_state.is_correct = True
        st.session_state.streak += 1
        st.session_state.score += 1
        st.session_state.error_msg = ""
        st.success("✅ शाबाश! सही जवाब!")
    else:
        # Reset streak, save feedback message, increment attempt_id, and rerun
        st.session_state.streak = 0
        st.session_state.error_msg = f"❌ फिर से कोशिश करो! आपने बोला: '{spoken_text}'. बोलो '{current_item['letter']}' या '{current_item['word']}'"
        st.session_state.attempt_id += 1
        st.rerun()

# Navigation Controls
st.markdown("---")
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

with btn_col1:
    if st.button("⬅️ पिछला (Previous)", disabled=st.session_state.idx == 0):
        st.session_state.idx -= 1
        st.session_state.is_correct = True
        st.session_state.error_msg = ""
        st.session_state.attempt_id = 0
        st.rerun()

with btn_col2:
    if st.button("गेम दोबारा शुरू करें (Restart) 🔄"):
        st.session_state.idx = 0
        st.session_state.streak = 0
        st.session_state.score = 0
        st.session_state.is_correct = False
        st.session_state.error_msg = ""
        st.session_state.attempt_id = 0
        st.session_state.start_time = time.time()
        st.session_state.reset_count += 1
        st.rerun()

with btn_col3:
    if st.button("अगला (Next) ➡️", disabled=not st.session_state.is_correct):
        if st.session_state.idx + 1 < len(ALPHABET):
            st.session_state.idx += 1
            st.session_state.is_correct = False
            st.session_state.error_msg = ""
            st.session_state.attempt_id = 0
            st.rerun()
        else:
            st.snow()
            st.title("🏆 GRAND FINALE TROPHY UNLOCKED! YOU WIN! 🏆")
