import datetime
from io import BytesIO
import subprocess
import threading
import requests
import logging
import wave
import time
import sys
import os
import signal
import traceback
import zipfile
from urllib.request import urlretrieve
import tempfile
from collections import deque

# KivyMD imports
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDButton, MDFabButton
from kivymd.uix.button.button import MDButtonText
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.dialog import MDDialog
from kivymd.uix.card import MDCard
from kivy.metrics import dp
from kivymd.uix.navigationrail import (
    MDNavigationRail,
    MDNavigationRailItem,
    MDNavigationRailItemIcon,
    MDNavigationRailItemLabel,
)
from kivy.clock import Clock
from kivy.properties import StringProperty

# Set clipboard provider to SDL2
import kivy

kivy.Config.set("kivy", "clipboard", "sdl2")

# Import required libraries
import pyaudio
# AGiXTSDK import is now optional:
try:
    from agixtsdk import AGiXTSDK
except ImportError:
    AGiXTSDK = None  # Set AGiXTSDK to None if import fails
    logging.warning("AGiXTSDK not found. AGiXT features will be disabled.")
from faster_whisper import WhisperModel
import webrtcvad
from pocketsphinx import Pocketsphinx, get_model_path
import numpy as np
from queue import Queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="agixt_listen.log",
)


class AGiXTListen:
    def __init__(
        self,
        server="http://localhost:7437",
        api_key="",
        agent_name="gpt4free",
        conversation_name="",
        whisper_model="base.en",
        wake_word="",
    ):
        self.server = server
        self.api_key = api_key
        self.agent_name = agent_name
        self.conversation_name = conversation_name or datetime.datetime.now().strftime(
            "%Y-%m-%d"
        )
        self.wake_word = wake_word.lower()
        self.wake_functions = {"chat": self.default_voice_chat}
        self.TRANSCRIPTION_MODEL = whisper_model
        self.audio = pyaudio.PyAudio()
        self.w = WhisperModel(
            self.TRANSCRIPTION_MODEL, download_root="models", device="cpu"
        )
        self.is_recording = False
        self.vad = webrtcvad.Vad(3)  # Aggressiveness is 3 (highest)
        self.is_speaking_activity = False
        self.sdk = None
        self.conversation_history = None
        self.transcribed_text = ""
        self.last_transcription_time = 0

        # New attributes for improved audio processing
        self.CHUNK = 480  # 30ms at 16kHz
        self.RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.audio_buffer = deque(maxlen=100)  # 3 seconds of audio at 16kHz
        self.vad_buffer = np.zeros(self.CHUNK, dtype=np.int16)

        # Minimum audio duration for transcription (in seconds)
        self.MIN_AUDIO_DURATION = 0.5

        # Queue for audio chunks to be transcribed
        self.transcription_queue = Queue()

        # Initialize SDK if API key is provided
        if self.api_key:
            self.initialize_sdk()

    def initialize_sdk(self):
        if AGiXTSDK is None:
            logging.warning(
                "AGiXTSDK is not installed. Skipping SDK initialization."
            )
            return False

        try:
            self.sdk = AGiXTSDK(base_uri=self.server, api_key=self.api_key)
            self.conversation_history = self.get_conversation_with_retry()
            return True
        except Exception as e:
            logging.error(f"Failed to initialize SDK: {str(e)}")
            return False

    def get_conversation_with_retry(self, max_retries=3, retry_delay=1):
        for attempt in range(max_retries):
            try:
                return self.sdk.get_conversation(
                    agent_name=self.agent_name,
                    conversation_name=self.conversation_name,
                    limit=20,
                    page=1,
                )
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"API call failed, retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logging.error(
                        f"Failed to get conversation after {max_retries} attempts: {str(e)}"
                    )
                    return None

    def continuous_record_and_transcribe(self):
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        logging.info("Starting continuous recording and transcription...")
        try:
            while self.is_recording:
                audio_chunk = stream.read(
                    self.CHUNK, exception_on_overflow=False
                )
                self.audio_buffer.append(audio_chunk)

                # Convert audio chunk to numpy array for VAD processing
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                self.vad_buffer = np.concatenate(
                    (self.vad_buffer[self.CHUNK :], audio_np)
                )

                try:
                    is_speech = self.vad.is_speech(
                        self.vad_buffer.tobytes(), self.RATE
                    )
                except Exception as e:
                    logging.error(f"VAD error: {str(e)}")
                    is_speech = False

                if is_speech:
                    self.last_transcription_time = time.time()

                current_time = time.time()
                if (
                    current_time - self.last_transcription_time
                    >= self.MIN_AUDIO_DURATION
                ):
                    self.process_audio_buffer()

        except Exception as e:
            logging.error(
                f"Error in continuous recording and transcription: {str(e)}"
            )
        finally:
            stream.stop_stream()
            stream.close()

    def process_audio_buffer(self):
        if len(self.audio_buffer) > 0:
            audio_data = b"".join(self.audio_buffer)
            self.transcription_queue.put(audio_data)
            self.audio_buffer.clear()
            self.last_transcription_time = time.time()

    def transcribe_worker(self):
        while True:
            audio_data = self.transcription_queue.get()
            if audio_data is None:
                break  # Exit the thread when None is received

            transcription = self.transcribe_audio(audio_data)
            if transcription:
                self.transcribed_text += transcription + " "
                MDApp.get_running_app().schedule_update_notes()

            self.transcription_queue.task_done()

    def transcribe_audio(self, audio_data):
        try:
            if not audio_data or len(audio_data) == 0:
                logging.warning(
                    "Empty audio data received. Skipping transcription."
                )
                return ""

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_wav:
                temp_wav_path = temp_wav.name
                with wave.open(temp_wav_path, "wb") as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(audio_data)

            segments, _ = self.w.transcribe(temp_wav_path)
            transcription = " ".join(segment.text for segment in segments)

            os.unlink(temp_wav_path)

            return transcription.strip()

        except Exception as e:
            logging.error(f"Error in transcribe_audio: {str(e)}")
            return ""

    def default_voice_chat(self, text):
        if not self.sdk:
            logging.warning("SDK not initialized. Cannot perform voice chat.")
            return

        logging.info(f"Sending text to agent: {text}")
        return self.sdk.chat(
            agent_name=self.agent_name,
            user_input=text,
            conversation=self.conversation_name,
            context_results=6,
        )

    def start_recording(self):
        if not self.sdk:
            logging.warning(
                "AGiXT SDK not initialized. Transcription will proceed without AGiXT."
            )

        self.is_recording = True
        self.input_recording_thread = threading.Thread(
            target=self.continuous_record_and_transcribe
        )
        self.conversation_check_thread = threading.Thread(
            target=self.check_conversation_updates
        )
        self.transcription_thread = threading.Thread(
            target=self.transcribe_worker
        )  # Start transcription worker thread
        self.input_recording_thread.start()
        self.conversation_check_thread.start()
        self.transcription_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.input_recording_thread:
            self.input_recording_thread.join()
        if self.conversation_check_thread:
            self.conversation_check_thread.join()
        self.transcription_queue.put(
            None
        )  # Signal transcription worker thread to exit
        self.transcription_thread.join()

    def check_conversation_updates(self):
        while self.is_recording:
            time.sleep(2)
            if self.sdk:
                new_history = self.get_conversation_with_retry()
                if new_history is None:
                    continue
                new_entries = [
                    entry
                    for entry in new_history
                    if entry not in self.conversation_history
                ]
                for entry in new_entries:
                    if entry.startswith("[ACTIVITY]"):
                        activity_message = entry.split("[ACTIVITY]")[
                            1
                        ].strip()
                        logging.info(
                            f"Received activity message: {activity_message}"
                        )
                        self.speak_activity(activity_message)
                self.conversation_history = new_history
            else:
                logging.info(
                    "AGiXT SDK not connected. Skipping conversation updates."
                )

    def speak_activity(self, message):
        self.is_speaking_activity = True
        self.text_to_speech(message)
        self.is_speaking_activity = False

    def text_to_speech(self, text):
        if not self.sdk:
            logging.warning(
                "SDK not initialized. Cannot perform text-to-speech."
            )
            return

        try:
            tts_url = self.sdk.text_to_speech(
                agent_name=self.agent_name, text=text
            )
            response = requests.get(tts_url)
            generated_audio = response.content
            stream = self.audio.open(
                format=pyaudio.paInt16, channels=1, rate=16000, output=True
            )
            stream.write(generated_audio)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logging.error(f"Error in text-to-speech conversion: {str(e)}")
            logging.debug(traceback.format_exc())

    def listen_for_wake_word(self):
        CHUNK = 480
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        logging.info(f"Listening for wake word: '{self.wake_word}'")

        if self.ps:
            while self.is_recording:
                frame = stream.read(CHUNK)
                is_speech = self.vad.is_speech(frame, RATE)

                if is_speech:
                    self.ps.start_utt()
                    self.ps.process_raw(frame, False, False)
                    if self.ps.hyp():
                        logging.info(f"Wake word detected: {self.wake_word}")
                        self.process_wake_word()
                    self.ps.end_utt()
        else:
            logging.warning(
                "Pocketsphinx not initialized. Cannot listen for wake word."
            )
        stream.stop_stream()
        stream.close()

    def process_wake_word(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        audio_data = b"".join(frames)
        transcription = self.transcribe_audio(audio_data)
        for wake_word, wake_function in self.wake_functions.items():
            if wake_word.lower() in transcription.lower():
                response = wake_function(transcription)
                if response:
                    self.text_to_speech(response)
                break
        else:
            response = self.default_voice_chat(transcription)
            if response:
                self.text_to_speech(response)


class CustomNavigationRailItem(MDNavigationRailItem):
    icon = StringProperty()
    text = StringProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add icon and label directly in the constructor
        if self.icon:
            self.add_widget(MDNavigationRailItemIcon(icon=self.icon))
        if self.text:
            self.add_widget(MDNavigationRailItemLabel(text=self.text))


class CustomNavigationRail(MDNavigationRail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "selected"
        self.pos_hint = {"top": 1}  # Position the rail at the top


class AGiXTNoteApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.listener = None
        self.is_recording = False
        self.settings_dialog = None
        self.current_view = "record"

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Teal"

        main_layout = MDBoxLayout(orientation="horizontal")

        # Navigation Rail
        self.nav_rail = self.create_navigation_rail()
        main_layout.add_widget(self.nav_rail)

        # Content Area
        self.content_area = MDBoxLayout(
            orientation="vertical", padding=dp(20), spacing=dp(20)
        )

        # Record View
        self.record_view = self.create_record_view()

        # Settings View
        self.settings_view = self.create_settings_view()

        # Initially show the record view
        self.content_area.add_widget(self.record_view)
        main_layout.add_widget(self.content_area)

        return main_layout

    def create_navigation_rail(self):
        nav_rail = CustomNavigationRail()

        record_item = MDNavigationRailItem(
            MDNavigationRailItemIcon(icon="microphone"),
            MDNavigationRailItemLabel(text="Record"),
            on_release=lambda x: self.switch_view("record"),
        )
        settings_item = MDNavigationRailItem(
            MDNavigationRailItemIcon(icon="cog"),
            MDNavigationRailItemLabel(text="Settings"),
            on_release=lambda x: self.switch_view("settings"),
        )

        nav_rail.add_widget(record_item)
        nav_rail.add_widget(settings_item)

        # Delay setting the anchor_button
        def set_anchor_button(dt):
            nav_rail.anchor_button = record_item

        Clock.schedule_once(set_anchor_button, 0.1)  # Delay of 0.1 seconds

        return nav_rail

    def create_record_view(self):
        record_view = MDBoxLayout(orientation="vertical", spacing=dp(20))

        # Notes Card
        notes_card = MDCard(
            orientation="vertical",
            padding=dp(10),
            size_hint=(1, 0.8),
            elevation=4,
            md_bg_color=self.theme_cls.surfaceColor,
        )
        scroll_view = MDScrollView(size_hint=(1, 1))
        self.notes_label = MDLabel(
            text="Notes will appear here...",
            halign="left",
            valign="top",
            size_hint_y=None,
            theme_text_color="Custom",
            text_color=self.theme_cls.onSurfaceColor,
        )
        self.notes_label.bind(texture_size=self.notes_label.setter("size"))
        scroll_view.add_widget(self.notes_label)
        notes_card.add_widget(scroll_view)
        record_view.add_widget(notes_card)

        # Controls Layout
        controls_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(60),
            padding=dp(10),
            spacing=dp(20),
        )

        self.start_button = MDFabButton(
            icon="microphone", style="standard", on_release=self.toggle_recording
        )
        controls_layout.add_widget(self.start_button)

        self.send_to_agixt_button = MDFabButton(
            icon="send",
            style="standard",
            on_release=self.send_transcription_to_agixt,
            disabled=True,
        )
        controls_layout.add_widget(self.send_to_agixt_button)

        record_view.add_widget(controls_layout)
        return record_view

    def create_settings_view(self):
        settings_view = MDBoxLayout(orientation="vertical", spacing=dp(20))
        self.server_input = MDTextField(
            hint_text="Server URL", text="http://localhost:7437"
        )
        self.api_key_input = MDTextField(hint_text="API Key", password=True)
        self.agent_name_input = MDTextField(hint_text="Agent Name", text="gpt4free")
        self.conversation_name_input = MDTextField(hint_text="Conversation Name")
        self.wake_word_input = MDTextField(hint_text="Wake Word (Optional)")

        settings_view.add_widget(self.server_input)
        settings_view.add_widget(self.api_key_input)
        settings_view.add_widget(self.agent_name_input)
        settings_view.add_widget(self.conversation_name_input)
        settings_view.add_widget(self.wake_word_input)

        save_button = MDButton(
            MDButtonText(text="Save Settings"),
            style="filled",
            on_release=self.save_settings,
        )
        settings_view.add_widget(save_button)
        return settings_view

    def switch_view(self, view):
        self.content_area.clear_widgets()
        if view == "record":
            self.content_area.add_widget(self.record_view)
        elif view == "settings":
            self.content_area.add_widget(self.settings_view)
        self.current_view = view

    def toggle_recording(self, instance):
        if not self.listener:
            self.show_error_dialog("Error", "Please configure settings first.")
            return

        if self.is_recording:
            self.is_recording = False
            self.listener.stop_recording()
            self.start_button.icon = "microphone"
            self.send_to_agixt_button.disabled = not self.listener.sdk
        else:
            self.is_recording = True
            self.listener.start_recording()
            self.start_button.icon = "stop"
            self.send_to_agixt_button.disabled = True

    def schedule_update_notes(self):
        Clock.schedule_once(self._update_notes)

    def _update_notes(self, *args):
        if self.listener:
            self.notes_label.text = self.listener.transcribed_text
            if hasattr(self.notes_label, "parent") and isinstance(
                self.notes_label.parent, MDScrollView
            ):
                self.notes_label.parent.scroll_y = 0

    def send_transcription_to_agixt(self, instance):
        if self.listener and self.listener.sdk and self.listener.transcribed_text:
            try:
                memory_text = (
                    f"Content of voice transcription:\n{self.listener.transcribed_text}"
                )
                self.listener.sdk.learn_text(
                    agent_name=self.listener.agent_name,
                    user_input=self.listener.transcribed_text,
                    text=memory_text,
                    collection_number=self.listener.conversation_name,
                )
                logging.info("Transcription sent to AGiXT.")
                self.show_error_dialog("Success", "Transcription sent to AGiXT.")
            except Exception as e:
                logging.error(f"Error sending transcription to AGiXT: {str(e)}")
                self.show_error_dialog(
                    "Error", f"Failed to send transcription to AGiXT: {str(e)}"
                )
        else:
            self.show_error_dialog(
                "Error", "AGiXT not connected or no transcription available."
            )

    def save_settings(self, instance):
        try:
            server = self.server_input.text
            api_key = self.api_key_input.text
            agent_name = self.agent_name_input.text
            conversation_name = self.conversation_name_input.text
            wake_word = self.wake_word_input.text

            if wake_word:
                self._download_pocketsphinx_model()

            self.listener = AGiXTListen(
                server=server,
                api_key=api_key,
                agent_name=agent_name,
                conversation_name=conversation_name,
                wake_word=wake_word,
            )

            if api_key:
                if not self.listener.initialize_sdk():
                    self.show_error_dialog(
                        "Error",
                        "Failed to initialize AGiXT SDK. Please check your settings.",
                    )
                    return

            if wake_word and self.listener.sdk:
                self.listener.ps = Pocketsphinx(
                    hmm=get_model_path("en-us"),
                    lm=False,
                    keyphrase=self.listener.wake_word,
                    kws_threshold=1e-20,
                )

            self.show_error_dialog("Success", "Settings saved successfully.")
            self.switch_view("record")
        except Exception as e:
            self.show_error_dialog("Error", f"Failed to save settings: {str(e)}")

    def _download_pocketsphinx_model(
        self,
        model_name="en-us",
        model_url="https://github.com/bambocher/pocketsphinx-schemas/raw/master/model/en-us.zip",
    ):
        model_path = get_model_path()
        model_dir = os.path.join(model_path, model_name)

        if not os.path.exists(model_dir):
            logging.info(
                f"Pocketsphinx model '{model_name}' not found. Downloading..."
            )
            try:
                temp_file, _ = urlretrieve(model_url)

                with zipfile.ZipFile(temp_file, "r") as zip_ref:
                    zip_ref.extractall(model_path)

                logging.info(
                    f"Pocketsphinx model '{model_name}' downloaded and extracted successfully."
                )
                os.remove(temp_file)
            except Exception as e:
                logging.error(
                    f"Error downloading or extracting Pocketsphinx model: {str(e)}"
                )
                raise

    def show_error_dialog(self, title, message):
        dialog = MDDialog(
            title=title,
            text=message,
            buttons=[
                MDButton(
                    MDButtonText(text="OK"),
                    style="text",
                    on_release=lambda x: dialog.dismiss(),
                )
            ],
        )
        dialog.open()


if __name__ == "__main__":
    app = AGiXTNoteApp()
    app.run()