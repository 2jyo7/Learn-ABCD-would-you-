import re
import time
import jellyfish
import librosa
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline

# Set Streamlit Page Configuration
st.set_page_config(page_title="🔤 Alphabet Adventure for Kids", layout="centered")

# Load speech recognition pipeline
@st.cache_resource
def load_speech_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

transcriber = load_speech_model()

# Alphabet items with integrated Letter + Object images
ALPHABET = [
    {"letter": "A", "word": "Apple", "image": "https://img.freepik.com/free-vector/letter-a-apple_1308-76815.jpg"},
    {"letter": "B", "word": "Ball", "image": "https://img.freepik.com/free-vector/letter-b-ball_1308-76822.jpg"},
    {"letter": "C", "word": "Cat", "image": "https://img.freepik.com/free-vector/letter-c-cat_1308-76829.jpg"},
    {"letter": "D", "word": "Dog", "image": "https://img.freepik.com/free-vector/letter-d-dog_1308-76836.jpg"},
    {"letter": "E", "word": "Elephant", "image": "https://img.freepik.com/free-vector/letter-e-elephant_1308-76843.jpg"},
    {"letter": "F", "word": "Fish", "image": "https://img.freepik.com/free-vector/letter-f-fish_1308-76850.jpg"},
    {"letter": "G", "word": "Grapes", "image": "https://img.freepik.com/free-vector/letter-g-grapes_1308-76857.jpg"},
    {"letter": "H", "word": "Hat", "image": "https://img.freepik.com/free-vector/letter-h-hat_1308-76864.jpg"},
    {"letter": "I", "word": "Ice cream", "image": "https://img.freepik.com/free-vector/letter-i-ice-cream_1308-76871.jpg"},
    {"letter": "J", "word": "Juice", "image": "https://img.freepik.com/free-vector/letter-j-juice_1308-76878.jpg"},
    {"letter": "K", "word": "Kite", "image": "https://img.freepik.com/free-vector/letter-k-kite_1308-76885.jpg"},
    {"letter": "L", "word": "Lion", "image": "https://img.freepik.com/free-vector/letter-l-lion_1308-76892.jpg"},
    {"letter": "M", "word": "Monkey", "image": "https://img.freepik.com/free-vector/letter-m-monkey_1308-76899.jpg"},
    {"letter": "N", "word": "Nest", "image": "https://img.freepik.com/free-vector/letter-n-nest_1308-76906.jpg"},
    {"letter": "O", "word": "Orange", "image": "https://img.freepik.com/free-vector/letter-o-orange_1308-76913.jpg"},
    {"letter": "P", "word": "Parrot", "image": "https://img.freepik.com/free-vector/letter-p-parrot_1308-76920.jpg"},
    {"letter": "Q", "word": "Queen", "image": "https://img.freepik.com/free-vector/letter-q-queen_1308-76927.jpg"},
    {"letter": "R", "word": "Rabbit", "image": "https://img.freepik.com/free-vector/letter-r-rabbit_1308-76934.jpg"},
    {"letter": "S", "word": "Sun", "image": "https://img.freepik.com/free-vector/letter-s-sun_1308-76941.jpg"},
    {"letter": "T", "word": "Tiger", "image": "https://img.freepik.com/free-vector/letter-t-tiger_1308-76948.jpg"},
    {"letter": "U", "word": "Umbrella", "image": "https://img.freepik.com/free-vector/letter-u-umbrella_1308-76955.jpg"},
    {"letter": "V", "word": "Van", "image": "https://img.freepik.com/free-vector/letter-v-van_1308-76962.jpg"},
    {"letter": "W", "word": "Watch", "image": "https://img.freepik.com/free-vector/letter-w-watch_1308-76969.jpg"},
    {"letter": "X", "word": "Xylophone", "image": "https://img.freepik.com/free-vector/letter-x-xylophone_1308-76976.jpg"},
    {"letter": "Y", "word": "Yak", "image": "https://img.freepik.com/free-vector/letter-y-yak_1308-76983.jpg"},
    {"letter": "Z", "word": "Zebra", "image": "https://img.freepik.com/free-vector/letter-z-zebra_1308-76990.jpg"}
]

LETTER_HOMOPHONES = {
    "A": ["a", "eh", "ay", "hey", "hay", "eight", "ei", "a4"],
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

SURPRISE_GIFTS = {
    5: "⭐ STAR BADGE UNLOCKED! Bright Super Star!",
    10: "🎈 BALLOONS UNLOCKED! Flying High!",
    15: "🎨 VIRTUAL CRAYONS UNLOCKED! Time to Color!",
    20: "👑 GOLDEN CROWN UNLOCKED! You are King/Queen of Alphabets!",
    25: "🚀 SPACE ROCKET BADGE UNLOCKED! Ready for Takeoff!"
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
if "surprise_gift" not in st.session_state:
    st.session_state.surprise_gift = ""

# Key name for current audio input
audio_key = f"mic_{st.session_state.idx}_{st.session_state.reset_count}_{st.session_state.attempt_id}"

# FIX FOR ERROR MESSAGE PERSISTENCE:
# Clear error_msg at top of script run if a new audio input is detected
if audio_key in st.session_state and st.session_state[audio_key] is not None:
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
        
        # Audio Player Component using Web Speech API
        phrase_to_say = f"{current_item['letter']} for {current_item['word']}"
        tts_code = f"""
        <button onclick="speak()" style="
            background-color: #FF4B4B;
            color: white;
            border: none;
            padding: 10px 18px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 5px;">
            🔊 सुनिए (Listen Pronunciation)
        </button>
        <script>
        function speak() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{phrase_to_say}');
            msg.rate = 0.65;  // Slower speech rate (0.1 to 10, 1 is normal)
            msg.pitch = 0.9;   // Warmer tone (0 to 2, 1 is default)
            msg.volume = 0.7;  // Softer volume (0 to 1)
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """
        components.html(tts_code, height=60)

# Display error message only when present
if st.session_state.error_msg:
    st.error(st.session_state.error_msg)

audio_file = st.audio_input("🎙️ माइक दबाकर अपनी आवाज रिकॉर्ड करें", key=audio_key)

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    
    audio_data, _ = librosa.load("temp_audio.wav", sr=16000)
    result = transcriber(audio_data)
    spoken_text = re.sub(r'[^\w\s]', '', result["text"].strip().lower())
    
    st.write(f"🗣️ **You said:** *'{spoken_text}'*")
    
    if validate_pronunciation(spoken_text, current_item["letter"], current_item["word"]):
        st.session_state.error_msg = ""
        st.session_state.is_correct = True
        st.session_state.streak += 1
        st.session_state.score += 1
        st.success("✅ शाबाश! सही जवाब!")

        if st.session_state.score in SURPRISE_GIFTS:
            st.session_state.surprise_gift = SURPRISE_GIFTS[st.session_state.score]
            st.balloons()
            
        if st.session_state.surprise_gift:
            st.info(f"🎁 **SURPRISE UNLOCKED:** {st.session_state.surprise_gift}")

    else:
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
        st.session_state.surprise_gift = ""
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
            st.balloons()
            st.title("🏆 GRAND FINALE TROPHY UNLOCKED! YOU WIN! 🏆")
            st.balloons()
