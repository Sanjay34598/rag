import sys
import io
import time
from pathlib import Path
from gtts import gTTS

# Configure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.stt.sarvam_stt import get_stt_service

def generate_hindi_mp3(text: str) -> bytes:
    tts = gTTS(text=text, lang='hi')
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()

def main():
    print("==================================================")
    print("SARVAM STT ACCURACY TEST (5 HINDI WORD TYPES)")
    print("==================================================")
    
    stt_service = get_stt_service()
    
    test_cases = [
        ("Common Hindi", "भारत की राजधानी क्या है?"),
        ("English Loanword", "कॉर्पोरेशन क्या है?"),
        ("Technical Term", "वायुमंडलीय दबाव की परिभाषा"),
        ("Proper Noun", "लिंकन शहर कहाँ स्थित है?"),
        ("Compound Phrase", "ईमानदारी और सच्चाई का महत्व")
    ]
    
    results = []
    
    for category, spoken_text in test_cases:
        audio_bytes = generate_hindi_mp3(spoken_text)
        success, res_dict, lat_ms = stt_service.transcribe(audio_bytes, filename="test.mp3", mime_type="audio/mp3")
        
        transcript = res_dict.get("transcript", "")
        exact_match = (spoken_text.replace("?", "").replace(".", "").strip() == transcript.replace("?", "").replace(".", "").strip())
        
        print(f"\nCategory    : {category}")
        print(f"Spoken Text : '{spoken_text}'")
        print(f"Transcript  : '{transcript}'")
        print(f"Exact Match : {'YES' if exact_match else 'NO'}")
        print(f"STT Latency : {lat_ms:.2f} ms")
        
        results.append({
            "category": category,
            "spoken": spoken_text,
            "transcript": transcript,
            "exact_match": exact_match,
            "lat_ms": lat_ms
        })

if __name__ == "__main__":
    main()
