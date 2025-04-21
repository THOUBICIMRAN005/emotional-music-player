import cv2
import numpy as np
import pygame
import time
import os
from random import choice
from collections import defaultdict
from fer import FER

# Initialize pygame mixer with modern settings
pygame.mixer.init()
pygame.mixer.pre_init(44100, -16, 2, 2048)  # Better audio configuration

# Music library - replace with your actual file paths
MUSIC_LIBRARY = {
    "happy": ["happy_music.mp3"],
    "sad": ["sad_music.mp3"],
    "angry": ["intense_music.mp3"],
    "neutral": ["calm_music.mp3"]
}

# Verify music files exist
for emotion, tracks in MUSIC_LIBRARY.items():
    for track in tracks:
        if not os.path.exists(track):
            print(f"Warning: Music file not found - {track}")

class EmotionMusicPlayer:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.cap = cv2.VideoCapture(0)
        self.current_emotion = "neutral"
        self.current_track = ""
        self.volume = 0.7
        self.emotion_history = []
        self.history_length = 5
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Emotion color mapping
        self.emotion_colors = {
            "happy": (0, 255, 255),  # Yellow
            "sad": (255, 0, 0),     # Blue
            "angry": (0, 0, 255),   # Red
            "neutral": (255, 255, 255)  # White
        }

    def get_random_track(self, emotion):
        """Select a random track for the detected emotion"""
        available = [t for t in MUSIC_LIBRARY.get(emotion, []) 
                    if os.path.exists(t) and t != self.current_track]
        return choice(available) if available else None

    def play_music(self, emotion):
        """Play music matching the detected emotion"""
        if emotion != self.current_emotion or not pygame.mixer.music.get_busy():
            track = self.get_random_track(emotion)
            if track:
                try:
                    pygame.mixer.music.load(track)
                    pygame.mixer.music.set_volume(self.volume)
                    pygame.mixer.music.play()
                    self.current_track = track
                    self.current_emotion = emotion
                    print(f"Playing: {os.path.basename(track)} for {emotion}")
                except pygame.error as e:
                    print(f"Audio error: {e}")

    def update_emotion_history(self, emotion, score):
        """Maintain emotion history for smoother transitions"""
        self.emotion_history.append((emotion, score))
        if len(self.emotion_history) > self.history_length:
            self.emotion_history.pop(0)

    def get_dominant_emotion(self):
        """Determine the most frequent recent emotion"""
        if not self.emotion_history:
            return "neutral"
        
        freq = defaultdict(int)
        for emotion, _ in self.emotion_history:
            freq[emotion] += 1
        
        return max(freq.items(), key=lambda x: x[1])[0]

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Detect emotions
                results = self.detector.detect_emotions(frame)
                
                if results:
                    emotion, score = self.detector.top_emotion(frame)
                    self.update_emotion_history(emotion, score)
                    dominant_emotion = self.get_dominant_emotion()
                    self.play_music(dominant_emotion)
                    
                    # Draw face box and emotion text
                    (x, y, w, h) = results[0]["box"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), 
                                self.emotion_colors.get(dominant_emotion, (255, 255, 255)), 2)
                    cv2.putText(frame, f"{dominant_emotion}: {score:.2f}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               self.emotion_colors.get(dominant_emotion, (255, 255, 255)), 2)
                
                # Display UI elements
                cv2.putText(frame, f"Now: {os.path.basename(self.current_track)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Volume: {int(self.volume*100)}%", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, "Q:Quit  Space:Pause  +/-:Volume  N:Next", 
                           (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Emotion Music Player", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.pause()
                    else:
                        pygame.mixer.music.unpause()
                elif key == ord('+'):
                    self.volume = min(1.0, self.volume + 0.1)
                    pygame.mixer.music.set_volume(self.volume)
                elif key == ord('-'):
                    self.volume = max(0.0, self.volume - 0.1)
                    pygame.mixer.music.set_volume(self.volume)
                elif key == ord('n'):
                    self.play_music(self.current_emotion)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()

if __name__ == "__main__":
    player = EmotionMusicPlayer()
    player.run()
