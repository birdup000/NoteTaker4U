import base64
import threading
import sys
import subprocess
import wave
import os
import uuid
from io import BytesIO
from datetime import datetime

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests
try:
    import pyaudio
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
    import pyaudio
try:
    import webrtcvad
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "webrtcvad"])
    import webrtcvad
try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np
try:
    from agixtsdk import AGiXTSDK
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "agixtsdk"])
    from agixtsdk import AGiXTSDK

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.popup import Popup
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.image import Image
    from kivy.core.window import Window
    from kivy.graphics import Color, Rectangle
    from kivy.app import App
    from kivy.uix.gridlayout import GridLayout
    from kivy.utils import get_color_from_hex
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kivy"])
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.popup import Popup
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.image import Image
    from kivy.core.window import Window
    from kivy.graphics import Color, Rectangle
    from kivy.uix.gridlayout import GridLayout
    from kivy.utils import get_color_from_hex

audio = pyaudio.PyAudio()


class NoteTaker4UApp(App):
    def build(self):
        self.title = "NoteTaker4U"
        self.load_config()
        self.sdk = AGiXTSDK(base_uri=self.server, api_key=self.api_key)
        self.agent_name = self.agent_name
        self.whisper_model = self.whisper_model
        self.conversation_name = datetime.now().strftime("%Y-%m-%d")
        self.layout = NoteTaker4ULayout(
            server=self.server,
            api_key=self.api_key,
            agent_name=self.agent_name,
            whisper_model=self.whisper_model,
        )
        return self.layout

    def on_start(self):
        pass

    def build_config(self, config):
        config.setdefaults(
            "AGiXT",
            {
                "server": "http://localhost:7437",
                "api_key": "",
                "agent_name": "gpt4free",
                "whisper_model": "",
            },
        )

    def load_config(self):
        self.config = App.get_running_app().config
        config_file = os.path.join(os.path.dirname(__file__), "notetaker4u.ini")
        if os.path.exists(config_file):
            self.config.read(config_file)
            self.server = self.config.get("AGiXT", "server")
            self.api_key = self.config.get("AGiXT", "api_key")
            self.agent_name = self.config.get("AGiXT", "agent_name")
            self.whisper_model = self.config.get("AGiXT", "whisper_model")
        else:
            print("Configuration file not found. Using default values.")
            self.server = "http://localhost:7437"
            self.api_key = ""
            self.agent_name = "gpt4free"
            self.whisper_model = ""


from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

class NoteTaker4ULayout(GridLayout):
    def __init__(self, server, api_key, agent_name, whisper_model, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2
        self.padding = 20
        self.spacing = 20

        with self.canvas.before:
            Color(get_color_from_hex("#f2f2f2"))
            Rectangle(pos=self.pos, size=self.size)

        self.transcript_label = Label(
            text="Transcript",
            size_hint=(0.2, 0.1),
            color=get_color_from_hex("#333333"),
            font_size=18,
            bold=True,
        )
        self.add_widget(self.transcript_label)

        self.transcript_text = TextInput(
            size_hint=(0.8, 0.4),
            readonly=False,
            background_color=get_color_from_hex("#ffffff"),
            foreground_color=get_color_from_hex("#333333"),
            font_size=14,
            padding=(10, 10),
        )
        self.add_widget(self.transcript_text)

        self.notes_label = Label(
            text="Notes",
            size_hint=(0.2, 0.1),
            color=get_color_from_hex("#333333"),
            font_size=18,
            bold=True,
        )
        self.add_widget(self.notes_label)

        self.notes_text = TextInput(
            size_hint=(0.8, 0.4),
            readonly=False,
            background_color=get_color_from_hex("#ffffff"),
            foreground_color=get_color_from_hex("#333333"),
            font_size=14,
            padding=(10, 10),
        )
        self.add_widget(self.notes_text)

        self.start_button = Button(
            text="Start Listening",
            size_hint=(0.4, 0.1),
            on_press=self.start_listening,
            background_color=get_color_from_hex("#33cc33"),
            color=get_color_from_hex("#ffffff"),
            font_size=16,
        )
        self.add_widget(self.start_button)

        self.save_button = Button(
            text="Save Notes",
            size_hint=(0.4, 0.1),
            on_press=self.save_notes,
            background_color=get_color_from_hex("#33cc33"),
            color=get_color_from_hex("#ffffff"),
            font_size=16,
        )
        self.add_widget(self.save_button)

        self.progress_bar = ProgressBar(
            size_hint=(0.8, 0.05),
            value=0,
        )
        self.progress_bar.canvas.before.clear()
        with self.progress_bar.canvas.before:
            Color(get_color_from_hex("#33cc33"))
            Rectangle(pos=self.progress_bar.pos, size=self.progress_bar.size)
        self.add_widget(self.progress_bar)

        self.listener = AGiXTListen(
            server=server,
            api_key=api_key,
            agent_name=agent_name,
            whisper_model=whisper_model,
            wake_functions={
                "transcribe": self.transcribe_audio,
                "generate notes": self.generate_notes,
            },
        )

    def start_listening(self, instance):
        self.listener.listen()
        self.progress_bar.value = 100

    def transcribe_audio(self, text):
        self.transcript_text.text = text
        return "Transcription complete. What would you like me to do next?"

    def generate_notes(self, text):
        notes = self.listener.sdk.execute_command(
            agent_name=self.listener.agent_name,
            command_name="Summarize Text",
            command_args={"text": text},
            conversation_name=self.listener.conversation_name,
        )
        self.notes_text.text = notes
        return "Notes generated. You can now save them."

    def save_notes(self, instance):
        notes = self.notes_text.text.strip()
        if notes:
            file_name = f"notes_{self.listener.conversation_name}.txt"
            with open(file_name, "w") as f:
                f.write(notes)
            print(f"Notes saved to {file_name}")
        else:
            print("No notes to save.")

class NoteTakerApp(App):
    def build(self):
        Window.clearcolor = get_color_from_hex("#f2f2f2")
        return NoteTaker4ULayout(
            server="your_server_url",
            api_key="your_api_key",
            agent_name="your_agent_name",
            whisper_model="your_whisper_model",
        )





class AGiXTListen:
    def __init__(
        self,
        server="http://localhost:7437",
        api_key="",
        agent_name="gpt4free",
        whisper_model="",
        wake_functions={},
    ):
        self.sdk = AGiXTSDK(base_uri=server, api_key=api_key)
        self.agent_name = agent_name
        self.wake_functions = (
            wake_functions
            if wake_functions != {}
            else {
                "chat": self.voice_chat,
                "instruct": self.voice_instruct,
            }
        )
        self.conversation_name = datetime.now().strftime("%Y-%m-%d")
        self.w = None
        if whisper_model != "":
            try:
                from whisper_cpp import Whisper
            except ImportError:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "whisper-cpp-pybind"]
                )
                try:
                    from whisper_cpp import Whisper
                except:
                    whisper_model = ""
            if whisper_model != "":
                whisper_model = whisper_model.lower()
                if whisper_model not in [
                    "tiny",
                    "tiny.en",
                    "base",
                    "base.en",
                    "small",
                    "small.en",
                    "medium",
                    "medium.en",
                    "large",
                    "large-v1",
                ]:
                    whisper_model = "base.en"
                os.makedirs(
                    os.path.join(os.getcwd(), "models", "whispercpp"), exist_ok=True
                )
                model_path = os.path.join(
                    os.getcwd(), "models", "whispercpp", f"ggml-{whisper_model}.bin"
                )
                if not os.path.exists(model_path):
                    r = requests.get(
                        f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{whisper_model}.bin",
                        allow_redirects=True,
                    )
                    open(model_path, "wb").write(r.content)
                self.w = Whisper(model_path=model_path)

    def process_audio_data(self, frames, rms_threshold=500):
        audio_data = b"".join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np**2))
        if rms > rms_threshold:
            buffer = BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b"".join(frames))
            wav_buffer = buffer.getvalue()
            base64_audio = base64.b64encode(wav_buffer).decode()
            thread = threading.Thread(
                target=self.transcribe_audio,
                args=(base64_audio),
            )
            thread.start()

    def transcribe_audio(self, base64_audio):
        if self.w:
            filename = f"{uuid.uuid4().hex}.wav"
            file_path = os.path.join(os.getcwd(), filename)
            if not os.path.exists(file_path):
                raise RuntimeError(f"Failed to load audio: {filename} does not exist.")
            self.w.transcribe(file_path)
            transcribed_text = self.w.output()
            os.remove(os.path.join(os.getcwd(), filename))
        else:
            transcribed_text = self.sdk.execute_command(
                agent_name=self.agent_name,
                command_name="Transcribe WAV Audio",
                command_args={"base64_audio": base64_audio},
                conversation_name="AGiXT Terminal",
            )
            transcribed_text = transcribed_text.replace("[BLANK_AUDIO]", "")
        for wake_word, wake_function in self.wake_functions.items():
            if wake_word.lower() in transcribed_text.lower():
                print("Wake word detected! Executing wake function...")
                if wake_function:
                    response = wake_function(transcribed_text)
                else:
                    response = self.voice_chat(text=transcribed_text)
                if response:
                    tts_response = self.sdk.execute_command(
                        agent_name=self.agent_name,
                        command_name="Translate Text to Speech",
                        command_args={
                            "text": response,
                        },
                        conversation_name=datetime.now().strftime("%Y-%m-%d"),
                    )
                    tts_response = tts_response.replace("#GENERATED_AUDIO:", "")
                    generated_audio = base64.b64decode(tts_response)
                    stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        output=True,
                    )
                    stream.write(generated_audio)
                    stream.stop_stream()
                    stream.close()

    def listen(self):
        print("Listening for wake word...")
        vad = webrtcvad.Vad(1)
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=320,
        )
        frames = []
        silence_frames = 0
        while True:
            data = stream.read(320)
            frames.append(data)
            is_speech = vad.is_speech(data, 16000)
            if not is_speech:
                silence_frames += 1
                if silence_frames > 1 * 16000 / 320:
                    self.process_audio_data(frames)
                    frames = []  # Clear frames after processing
                    silence_frames = 0
            else:
                silence_frames = 0

    def voice_chat(self, text):
        print(f"Sending text to agent: {text}")
        text_response = self.sdk.chat(
            agent_name=self.agent_name,
            user_input=text,
            conversation=self.conversation_name,
            context_results=6,
        )
        return text_response

    def voice_instruct(self, text):
        print(f"Sending text to agent: {text}")
        text_response = self.sdk.instruct(
            agent_name=self.agent_name,
            user_input=text,
            conversation=self.conversation_name,
        )
        return text_response




if __name__ == "__main__":
    NoteTakerApp().run()

