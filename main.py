from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from cv_pipeline import VisionGoalPipeline
from audio_feedback import speak
import cv2

class VisionGoalApp(App):
    def build(self):
        self.pipeline = VisionGoalPipeline()
        self.img = Image()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.img

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret: return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.pipeline.detect_objects(frame_rgb)

        # Audio feedback
        for d in detections:
            if d['label'] == 'ball' and d['conf'] > 0.8:
                speak("Ball detected at center")
            elif 'player' in d['label']:
                speak(f"{d['label']} moving left")

        # Draw boxes
        for d in detections:
            x1, y1, x2, y2 = d['box']
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_rgb, f"{d['label']} {d['conf']:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        buf = frame_rgb.tobytes()
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img.texture = texture

if __name__ == '__main__':
    VisionGoalApp().run()
