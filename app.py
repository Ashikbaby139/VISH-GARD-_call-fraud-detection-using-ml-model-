from flask import Flask, render_template, jsonify, request
from predictor import predict_fraud_hybrid
from phone_checker import check_phone_number

import stt_vosk
import os
import json
from datetime import datetime
from collections import deque

LOG_FILE = "transcripts.log"
RECENT_HISTORY_LIMIT = 12

app = Flask(__name__)

# ---------- GLOBAL STATE ----------
fraud_saved = False   # 🔒 prevents duplicate evidence
detection_history = deque(maxlen=RECENT_HISTORY_LIMIT)
last_live_recorded_final = ""
_log_cache = {
    "stamp": None,
    "entries": [],
}


def _log_file_stamp():
    try:
        stats = os.stat(LOG_FILE)
    except FileNotFoundError:
        return None

    return (stats.st_mtime_ns, stats.st_size)


def _load_log_entries():
    stamp = _log_file_stamp()
    if stamp == _log_cache["stamp"]:
        return _log_cache["entries"]

    entries = []

    if stamp is not None:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    _log_cache["stamp"] = stamp
    _log_cache["entries"] = entries
    return entries


def load_recent_fraud_history():
    history = []

    for entry in _load_log_entries()[-RECENT_HISTORY_LIMIT:]:
        history.append({
            "time": entry.get("time", ""),
            "source": "log",
            "text": entry.get("text", ""),
            "decision": 0,
            "confidence": float(entry.get("confidence", 0) or 0),
            "keywords": entry.get("keywords", []),
        })

    return history


def record_detection(source, text, decision, confidence, keywords):
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "text": text.strip(),
        "decision": int(decision),
        "confidence": float(confidence),
        "keywords": keywords,
    }
    detection_history.append(entry)
    return entry


def get_dashboard_history():
    if detection_history:
        return list(detection_history)

    return load_recent_fraud_history()

# ---------- HOME ----------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard_data")
def dashboard_data():
    history = get_dashboard_history()
    latest = history[-1] if history else None

    return jsonify({
        "history": history,
        "latest": latest,
    })

# ---------- AUDIO PAGE ----------
@app.route("/audio")
def audio_page():
    return render_template("audio.html")


@app.route("/audio_upload")
def audio_upload_page():
    return render_template("audio_upload.html")
#-----------phone number page------

@app.route("/number")
def number_page():
    return render_template("number.html")


# ---------- LIVE LISTEN ----------
@app.route("/listen")
def listen_route():
    global fraud_saved, last_live_recorded_final

    # ✅ CORRECT CALL
    try:
        snapshot = stt_vosk.get_live_snapshot()
    except RuntimeError as exc:
        return jsonify({
            "text": "",
            "result": "Audio backend unavailable",
            "confidence": "",
            "keywords": [],
            "error": str(exc)
        }), 503

    text = snapshot.get("text", "")
    final_text = snapshot.get("final", "")

    if not text:
        return jsonify({
            "text": "",
            "result": "",
            "confidence": "",
            "keywords": []
        })

    decision, confidence, keywords = predict_fraud_hybrid(text)

    if final_text and final_text != last_live_recorded_final:
        record_detection("audio", text, decision, confidence, keywords)
        last_live_recorded_final = final_text

    result = "🚨 Scam Detected" if decision == 0 else "✅ No Scam"

    # 🔒 SAVE ONLY ONCE PER CALL
    if decision == 0 and not fraud_saved:
        stt_vosk.save_transcript_to_file(
            text=text,
            decision=decision,
            confidence=confidence,
            keywords=keywords
        )
        fraud_saved = True

    return jsonify({
        "text": text,
        "result": result,
        "confidence": f"{confidence} %",
        "keywords": keywords
    })


@app.route("/audio_status")
def audio_status():
    return jsonify(stt_vosk.get_listener_status())

# ---------- RESET ----------
@app.route("/reset")
def reset():
    global fraud_saved, last_live_recorded_final
    fraud_saved = False
    last_live_recorded_final = ""

    # ✅ RESET MODULE STATE CORRECTLY
    stt_vosk.reset_live_text()

    return jsonify({"status": "reset"})

# ---------- TEXT PAGE ----------
@app.route("/text")
def text_page():
    return render_template("text.html")

# ---------- TEXT ANALYSIS ----------
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    text = request.json.get("text", "").strip()

    decision, confidence, keywords = predict_fraud_hybrid(text)
    record_detection("text", text, decision, confidence, keywords)

    if decision == 0:
        stt_vosk.save_transcript_to_file(
            text=text,
            decision=decision,
            confidence=confidence,
            keywords=keywords
        )

    return jsonify({
        "text": text,
        "result": "🚨 Scam Detected" if decision == 0 else "✅ No Scam",
        "confidence": f"{confidence} %",
        "keywords": keywords
    })

@app.route("/analyze_audio_upload", methods=["POST"])
def analyze_audio_upload():
    uploaded_file = request.files.get("audio")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No audio file uploaded"}), 400

    try:
        transcript = stt_vosk.transcribe_uploaded_audio(uploaded_file)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400

    if not transcript:
        return jsonify({
            "transcript": "",
            "result": "No speech detected",
            "confidence": "0 %",
            "keywords": []
        })

    decision, confidence, keywords = predict_fraud_hybrid(transcript)
    record_detection("upload", transcript, decision, confidence, keywords)

    if decision == 0:
        stt_vosk.save_transcript_to_file(
            text=transcript,
            decision=decision,
            confidence=confidence,
            keywords=keywords
        )

    return jsonify({
        "transcript": transcript,
        "result": "ðŸš¨ Scam Detected" if decision == 0 else "âœ… No Scam",
        "confidence": f"{confidence} %",
        "keywords": keywords
    })

# ---------- LOG PAGE ----------
@app.route("/logs")
def logs_page():
    return render_template("logs.html", logs=_load_log_entries())

# ---------- DELETE SINGLE LOG ----------
@app.route("/delete_log", methods=["POST"])
def delete_log():
    index = int(request.json["index"])

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if 0 <= index < len(lines):
        lines.pop(index)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    _log_cache["stamp"] = None

    return jsonify({"status": "deleted"})

# ---------- CLEAR ALL LOGS ----------
@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    open(LOG_FILE, "w").close()
    _log_cache["stamp"] = None
    return jsonify({"status": "cleared"})

#---------phone number checking--------

@app.route("/check_number", methods=["POST"])
def check_number():
    number = request.json.get("number", "").strip()

    if not number:
        return jsonify({"error": "No number provided"})

    result = check_phone_number(number)
    return jsonify(result)

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
