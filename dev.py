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
import tarfile
from urllib.request import urlretrieve
import tempfile

# KivyMD imports
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.scrollview import ScrollView
from kivymd.uix.dialog import MDDialog
from kivymd.uix.card import MDCard
from kivy.uix.floatlayout import FloatLayout
from kivy.metrics import dp
from kivymd.uix.gridlayout import MDGridLayout
from kivy.clock import Clock
from kivymd.uix.toolbar import MDTopAppBar  # Correct import!
from kivymd.uix.list import OneLineListItem

# Set clipboard provider to SDL2
import kivy
kivy.Config.set('kivy', 'clipboard', 'sdl2')

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
        self.input_recording_thread = None
        self.output_recording_thread = None
        self.wake_word_thread = None
        self.conversation_check_thread = None
        self.vad = webrtcvad.Vad(3)  # Aggressiveness is 3 (highest)
        self.ps = None
        self.is_speaking_activity = False
        self.sdk = None
        self.conversation_history = None
        self.transcribed_text = ""
        self.last_transcription_time = 0  # Add timestamp for silence detection

        # Initialize SDK if API key is provided
        if self.api_key:
            self.initialize_sdk()

    def initialize_sdk(self):
        if AGiXTSDK is None:  # Check if AGiXTSDK is available
            logging.warning(
                "AGiXTSDK is not installed. Skipping SDK initialization."
            )
            return False

        try:
            self.sdk = AGiXTSDK(base_uri=self.server, api_key=self.api_key)
            self.conversation_history = self.sdk.get_conversation(
                agent_name=self.agent_name,
                conversation_name=self.conversation_name,
                limit=20,
                page=1,
            )
            return True  # Indicate successful initialization
        except Exception as e:
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response.status_code == 401
            ):
                logging.error(
                    f"Failed to initialize SDK: Authentication Error - {str(e)}"
                )
                MDApp.get_running_app().show_error_dialog(
                    "Authentication Error",
                    "Invalid API key or unauthorized access. Please check your settings.",
                )
            else:
                logging.error(f"Failed to initialize SDK: {str(e)}")
                MDApp.get_running_app().show_error_dialog(
                    "SDK Initialization Error",
                    f"Failed to initialize the AGiXT SDK: {str(e)}",
                )
            self.sdk = None
            self.conversation_history = None
            return False  # Indicate failed initialization

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

    def transcribe_audio(self, audio_data, translate=False):
        try:
            # Check if audio_data is empty
            if not audio_data or len(audio_data) == 0:
                logging.warning(
                    "Empty audio data received. Skipping transcription."
                )
                return ""

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_wav:
                temp_wav_path = temp_wav.name
                with wave.open(temp_wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)

            # Transcribe the temporary WAV file
            segments, _ = self.w.transcribe(temp_wav_path)
            transcription = " ".join(segment.text for segment in segments)

            # Remove the temporary file
            os.unlink(temp_wav_path)

            return transcription

        except Exception as e:
            logging.error(f"Error in transcribe_audio: {str(e)}")
            logging.debug(traceback.format_exc())
            return ""



    def continuous_record_and_transcribe(self, is_input):
        CHUNK = 480  # 30ms at 16kHz
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 5  # Seconds of silence before sending transcription

        if is_input:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
        else:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK,
            )

        audio_type = "input" if is_input else "output"
        logging.info(
            f"Starting continuous recording and transcription for {audio_type} audio..."
        )
        try:
            buffer = b""  # Buffer to accumulate audio chunks
            while self.is_recording:
                if is_input:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    buffer += data
                    try:
                        is_speech = self.vad.is_speech(data, RATE)
                    except Exception as e:
                        logging.error(f"VAD error: {str(e)}")
                        is_speech = False  # Assume it's not speech if VAD fails

                    if is_speech:
                        self.last_transcription_time = time.time()

                    # Check for silence and send accumulated transcription
                    if (
                        time.time() - self.last_transcription_time
                        > SILENCE_THRESHOLD
                        and len(buffer) > 0
                    ):
                        transcription = self.transcribe_audio(buffer)
                        buffer = b""  # Reset buffer after sending

                        if len(transcription) > 10:
                            start_time = datetime.datetime.now()
                            memory_text = f"Content of {audio_type} voice transcription from {start_time}:\n{transcription}"
                            if self.sdk:
                                self.sdk.learn_text(
                                    agent_name=self.agent_name,
                                    user_input=transcription,
                                    text=memory_text,
                                    collection_number=self.conversation_name,
                                )
                                logging.info(
                                    f"Saved {audio_type} transcription to agent memories: {transcription}"
                                )
                            else:
                                logging.info(
                                    f"AGiXT SDK not connected. Transcription not sent: {transcription}"
                                )

                            self.transcribed_text += transcription + " "
                            Clock.schedule_once(
                                MDApp.get_running_app()._update_notes
                            )  # Schedule UI update
                else:
                    # For output stream, we don't need to read data
                    if not self.is_speaking_activity:
                        # Write silence when not speaking
                        stream.write(b"\x00" * CHUNK)
                    else:
                        # When speaking, we assume the data is provided elsewhere
                        time.sleep(CHUNK / RATE)  # Sleep for the duration of one chunk

        except Exception as e:
            logging.error(
                f"Error in continuous recording and transcription: {str(e)}"
            )
            logging.debug(traceback.format_exc())
        finally:
            stream.stop_stream()
            stream.close()

    def start_recording(self):
        if not self.sdk:
            logging.warning(
                "AGiXT SDK not initialized. Transcription will proceed without AGiXT."
            )

        self.is_recording = True
        self.input_recording_thread = threading.Thread(
            target=self.continuous_record_and_transcribe, args=(True,)
        )
        self.output_recording_thread = threading.Thread(
            target=self.continuous_record_and_transcribe, args=(False,)
        )
        self.conversation_check_thread = threading.Thread(
            target=self.check_conversation_updates
        )

        # Start wake word thread only if wake_word is set
        if self.wake_word and self.ps:
            self.wake_word_thread = threading.Thread(
                target=self.listen_for_wake_word
            )
            self.wake_word_thread.start()

        self.input_recording_thread.start()
        self.output_recording_thread.start()
        self.conversation_check_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.input_recording_thread:
            self.input_recording_thread.join()
        if self.output_recording_thread:
            self.output_recording_thread.join()
        if self.wake_word_thread:
            self.wake_word_thread.join()
        if self.conversation_check_thread:
            self.conversation_check_thread.join()

    def check_conversation_updates(self):
        while self.is_recording:
            time.sleep(2)  # Check every 2 seconds
            if self.sdk:
                new_history = self.sdk.get_conversation(
                    agent_name=self.agent_name,
                    conversation_name=self.conversation_name,
                    limit=20,
                    page=1,
                )
                new_entries = [
                    entry
                    for entry in new_history
                    if entry not in self.conversation_history
                ]
                for entry in new_entries:
                    if entry.startswith("[ACTIVITY]"):
                        activity_message = entry.split("[ACTIVITY]")[1].strip()
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
        CHUNK = 480  # 30ms at 16kHz
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

        # Check if Pocketsphinx is initialized
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


class AGiXTNoteApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.listener = None
        self.is_recording = False
        self.settings_dialog = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Teal"

        # Main Screen
        screen = MDScreen()

        # Toolbar
        toolbar = MDTopAppBar(title="AGiXT Advanced Notes")  # Use MDTopAppBar
        toolbar.right_action_items = [
            ["settings", lambda x: self.open_settings_dialog()]
        ]
        screen.add_widget(toolbar)

        # Main Layout
        main_layout = MDBoxLayout(
            orientation="vertical", padding=dp(20), spacing=dp(20)
        )

        # Notes Area (Scrollable Card)
        notes_card = MDCard(
            orientation="vertical",
            padding=dp(10),
            size_hint=(1, 0.8),  # Adjusted size_hint
            elevation=4,
        )
        self.notes_label = MDLabel(
            text="Notes will appear here...", halign="left", valign="top"
        )
        notes_card.add_widget(self.notes_label)
        main_layout.add_widget(notes_card)

        # Controls Area
        controls_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(60),
            padding=dp(10),
            spacing=dp(20),
        )

        self.start_button = MDRaisedButton(text="Start Recording")
        self.start_button.bind(on_release=self.toggle_recording)
        controls_layout.add_widget(self.start_button)

        self.send_to_agixt_button = MDRaisedButton(
            text="Send to AGiXT", disabled=True
        )
        self.send_to_agixt_button.bind(
            on_release=self.send_transcription_to_agixt
        )
        controls_layout.add_widget(self.send_to_agixt_button)

        main_layout.add_widget(controls_layout)
        screen.add_widget(main_layout)
        return screen

    def toggle_recording(self, instance):
        if not self.listener:
            self.show_error_dialog("Error", "Please configure settings first.")
            return

        if self.is_recording:
            self.is_recording = False
            self.listener.stop_recording()
            self.start_button.text = "Start Recording"
            self.send_to_agixt_button.disabled = (
                not self.listener.sdk
            )  # Enable button only if AGiXT is configured
        else:
            self.is_recording = True
            self.listener.start_recording()
            self.start_button.text = "Stop Recording"
            self.send_to_agixt_button.disabled = True

    def _update_notes(self, dt):  # Helper function for UI update
        if self.listener:
            self.notes_label.text = (
                self.listener.transcribed_text
            )  # Update label text

    def send_transcription_to_agixt(self, instance):
        if (
            self.listener
            and self.listener.sdk
            and self.listener.transcribed_text
        ):
            try:
                memory_text = f"Content of voice transcription:\n{self.listener.transcribed_text}"
                self.listener.sdk.learn_text(
                    agent_name=self.listener.agent_name,
                    user_input=self.listener.transcribed_text,
                    text=memory_text,
                    collection_number=self.listener.conversation_name,
                )
                logging.info("Transcription sent to AGiXT.")
                self.show_error_dialog(
                    "Success", "Transcription sent to AGiXT."
                )
            except Exception as e:
                logging.error(
                    f"Error sending transcription to AGiXT: {str(e)}"
                )
                self.show_error_dialog(
                    "Error",
                    f"Failed to send transcription to AGiXT: {str(e)}",
                )
        else:
            self.show_error_dialog(
                "Error", "AGiXT not connected or no transcription available."
            )

    def open_settings_dialog(self):
        if self.settings_dialog:
            self.settings_dialog.open()
            return

        # Improved Settings Dialog UI
        dialog_content = MDBoxLayout(
            orientation="vertical",
            spacing="12dp",
            size_hint_y=None,
            height="300dp",
        )
        self.server_input = MDTextField(
            hint_text="Server URL", text="http://localhost:7437"
        )
        self.api_key_input = MDTextField(
            hint_text="API Key", password=True
        )
        self.agent_name_input = MDTextField(
            hint_text="Agent Name", text="gpt4free"
        )
        self.conversation_name_input = MDTextField(
            hint_text="Conversation Name"
        )
        self.wake_word_input = MDTextField(
            hint_text="Wake Word (Optional)"
        )

        dialog_content.add_widget(self.server_input)
        dialog_content.add_widget(self.api_key_input)
        dialog_content.add_widget(self.agent_name_input)
        dialog_content.add_widget(self.conversation_name_input)
        dialog_content.add_widget(self.wake_word_input)

        self.settings_dialog = MDDialog(
            title="Settings",
            type="custom",
            content_cls=dialog_content,
            buttons=[
                MDFlatButton(
                    text="Cancel", on_release=self.close_settings_dialog
                ),
                MDRaisedButton(text="Save", on_release=self.save_settings),
            ],
        )
        self.settings_dialog.open()

    def save_settings(self, instance):
        try:
            server = self.server_input.text
            api_key = self.api_key_input.text
            agent_name = self.agent_name_input.text
            conversation_name = self.conversation_name_input.text
            wake_word = self.wake_word_input.text

            # Download Pocketsphinx model only if wake_word is provided
            if wake_word:
                self._download_pocketsphinx_model()

            # Initialize AGiXTListen
            self.listener = AGiXTListen(
                server=server,
                api_key=api_key,
                agent_name=agent_name,
                conversation_name=conversation_name,
                wake_word=wake_word,
            )

            # Only attempt to initialize the SDK if API key is provided
            if api_key:
                if not self.listener.initialize_sdk():
                    self.show_error_dialog(
                        "Error",
                        "Failed to initialize AGiXT SDK. Please check your settings.",
                    )
                    return  # Don't proceed if SDK initialization fails

            # Initialize Pocketsphinx if wake_word is set and listener is initialized
            if wake_word and self.listener.sdk:
                self.listener.ps = Pocketsphinx(
                    hmm=get_model_path("en-us"),
                    lm=False,
                    keyphrase=self.listener.wake_word,
                    kws_threshold=1e-20,
                )

            self.close_settings_dialog()
        except Exception as e:
            self.show_error_dialog(
                "Error", f"Failed to save settings: {str(e)}"
            )

    def _download_pocketsphinx_model(
        self,
        model_name="en-us",
        model_url="https://cmusphinx.github.io/wiki/download/pocketsphinx-en-us.tar.gz",
    ):
        """Downloads and extracts the Pocketsphinx model if it doesn't exist."""
        model_path = get_model_path()
        model_dir = os.path.join(model_path, model_name)

        if not os.path.exists(model_dir):
            logging.info(
                f"Pocketsphinx model '{model_name}' not found. Downloading..."
            )
            try:
                temp_file, _ = urlretrieve(model_url)

                with tarfile.open(temp_file, "r:gz") as tar:
                    for member in tar.getmembers():
                        tar.extract(member, model_path)
                        os.chmod(
                            os.path.join(model_path, member.name), 0o444
                        )

                logging.info(
                    f"Pocketsphinx model '{model_name}' downloaded and extracted successfully."
                )
                os.remove(temp_file)
            except Exception as e:
                logging.error(
                    f"Error downloading or extracting Pocketsphinx model: {str(e)}"
                )
                raise

    def close_settings_dialog(self, *args):
        if self.settings_dialog:
            self.settings_dialog.dismiss()

    def show_error_dialog(self, title, message):
        dialog = MDDialog(title=title, text=message)
        dialog.open()


if __name__ == "__main__":
    app = AGiXTNoteApp()
    app.run()