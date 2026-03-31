import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB — chunks are ~18 MB each

ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "flac"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def fmt_time(secs):
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    api_key = request.form.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "No OpenAI API key provided"}), 400

    language = request.form.get("language") or None
    prompt = request.form.get("prompt") or None

    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        client = OpenAI(api_key=api_key)
        with open(tmp_path, "rb") as audio_file:
            kwargs = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            result = client.audio.transcriptions.create(**kwargs)

        segments = []
        if hasattr(result, "segments") and result.segments:
            for s in result.segments:
                # Segments may be dicts or objects depending on the SDK version
                def g(key):
                    return s[key] if isinstance(s, dict) else getattr(s, key)
                segments.append({
                    "id": g("id"),
                    "start": round(g("start"), 2),
                    "end": round(g("end"), 2),
                    "start_fmt": fmt_time(g("start")),
                    "end_fmt": fmt_time(g("end")),
                    "text": g("text").strip(),
                })

        return jsonify({
            "text": result.text,
            "language": getattr(result, "language", None),
            "duration": getattr(result, "duration", None),
            "segments": segments,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/diarize", methods=["POST"])
def diarize():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    segments = data.get("segments", [])
    api_key = data.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "No OpenAI API key provided"}), 400
    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    # Build a numbered segment list for the prompt
    lines = []
    for s in segments:
        lines.append(f"[{s['id']}] ({s['start_fmt']}) {s['text']}")
    segments_block = "\n".join(lines)

    system_prompt = (
        "You are an expert speaker diarization assistant. "
        "Given a numbered list of transcript segments, determine which speaker said each segment. "
        "Use consistent labels like 'Speaker 1', 'Speaker 2', etc. "
        "Base your decisions on conversational turn-taking, question/answer patterns, topic shifts, and linguistic style. "
        "Return ONLY a valid JSON object with a single key 'speakers', whose value is an array of objects each with "
        "'id' (integer, matching the segment id) and 'speaker' (string label). "
        "Do not include any explanation or markdown."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript segments:\n\n{segments_block}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
