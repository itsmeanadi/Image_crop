import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from main import FingerFrameCamera, Config


class Klix:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Klix")
        self.app.geometry("1000x700")

        self.camera = FingerFrameCamera(Config())
        self.camera.cap = cv2.VideoCapture(0)
        self.running = True

        self.title = ctk.CTkLabel(
            self.app,
            text="Klix",
            font=("Arial", 32, "bold")
        )
        self.title.pack(pady=20)

        self.video = ctk.CTkLabel(self.app, text="")
        self.video.pack(padx=20, pady=10, fill="both", expand=True)

        self.button = ctk.CTkButton(
            self.app,
            text="Camera Running",
            width=200,
            height=50
        )
        self.button.pack(pady=20)

        self.app.protocol("WM_DELETE_WINDOW", self.close)
        self.update_camera()
        self.app.mainloop()

    def update_camera(self):
        if not self.running:
            return

        success, frame = self.camera.cap.read()

        if success:
            frame = cv2.flip(frame, 1)
            frame = self.camera._process_frame(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((900, 550))

            photo = ImageTk.PhotoImage(image)

            self.video.configure(image=photo)
            self.video.image = photo

        self.app.after(10, self.update_camera)

    def close(self):
        self.running = False

        if self.camera.cap:
            self.camera.cap.release()

        self.camera.hands.close()
        self.app.destroy()


Klix()