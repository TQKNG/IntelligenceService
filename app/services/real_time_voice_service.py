import aiohttp
import assemblyai as aai
from elevenlabs import  stream
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
aai_api_key = os.getenv("AAI_API_KEY")
elevenlab_api_key = os.getenv("ELEVENLAB_API_KEY")


class AI_Assistant:
    # Step 1: Setting configuration for sub-services and define the role of the assistant
    def __init__(self):
        # AssemblyAI API key
        aai.settings.api_key = aai_api_key 
        
        # OpenAI API key    
        self.openai_client = OpenAI(api_key=openai_api_key)

        # ElevenLabs API key
        self.elevenlabs_api_key = elevenlab_api_key

        self.transriber = None

        # Prompt
        self.full_transcript = [{
            "role":"system", "content":"You are a SQL Server Expert. Be precise and helpful. Limit your answer in one sentence"
        }]

    # Step 2: Real-Time Transcription with AssemblyAI/ Speech-to-Text
    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate = 16000,# number of audio samples per second (Hz),Higher sample rates result in higher quality audio
            
            on_data=self.on_data,
            on_error = self.on_error,
            on_open= self.on_open,
            on_close = self.on_close,
            end_utterance_silence_threshold=1000 # silent config
        )

        self.transcriber.connect()

        # Stream config: onnect microphone and stream voice
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(microphone_stream)


    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None


    def on_open(self, session_opened:aai.RealtimeTranscript):
        return

    # Transcript handler
    def on_data(self,transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        
        if isinstance(transcript,aai.RealtimeFinalTranscript):
          self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\n")

    def on_error(self, error: aai.RealtimeError):
        # print("An error occurred:",error)
        return

    def on_close(self):
        # print("Connection closed")
        return

 
     # Step 3: Generate AI response
    def generate_ai_response(self,transcript):
        self.stop_transcription()
        self.full_transcript.append({"role":"user", "content": transcript.text})
        print(f"\nUser: {transcript.text}", end="\r\n")

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=self.full_transcript
        )

        print("test response", response)    

        ai_response = response.choices[0].message.content

        self.generate_audio(ai_response)

        self.start_transcription()

    # Step 4: Generate Audio with ElevenLabs
    def generate_audio(self,text):
        self.full_transcript.append({"role":"assistant","content":text})
        print(f"\nAssistant: {text}", end="\r\n")

        audio_stream = self.generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice="Brian",
            stream=True
        )

        stream(audio_stream)
    

    def audio_to_text(self, path):
        self.transcriber = aai.Transcriber()
        transcript = self.transcriber.transcribe(path)

        if transcript.status == aai.TranscriptStatus.error:
            print(transcript.error)
        else:
            print(transcript.text)

# greeting = "Thank you for using Virbrix Analytic assistant. My name is Virbrix. How can I help you today?"

# ai_assistant = AI_Assistant()
# ai_assistant.generate_audio(greeting)
# ai_assistant.start_transcription()