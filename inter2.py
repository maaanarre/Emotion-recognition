import tkinter as tk
from tkinter import Button, Label, messagebox, Frame
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.cap = None
        self.running = False
        self.emotion_model = self.load_emotion_model()
        self.classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Initialize UI
        self.setup_ui()

    def load_emotion_model(self):
        try:
            with open("zakaria_model.json", "r") as json_file:
                loaded_model_json = json_file.read()
            emotion_model = model_from_json(loaded_model_json)
            emotion_model.load_weights("zakaria_model.weights.h5")
            print("Modèle de reconnaissance d'émotions chargé avec succès.")
            return emotion_model
        except Exception as e:
            print(f"Erreur de chargement du modèle: {str(e)}")  # Debug print
            messagebox.showerror("Erreur", f"Impossible de charger le modèle : {str(e)}")
            return None

    def preprocess_face(self, face):
        face_resized = cv2.resize(face, (48, 48))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=[0, -1])
        return face_resized

    def recognize_emotions(self, frame):
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                face = gray_frame[y:y + h, x:x + w]
                preprocessed_face = self.preprocess_face(face)

                if self.emotion_model:
                    predictions = self.emotion_model.predict(preprocessed_face)
                    emotion_index = np.argmax(predictions)
                    emotion_label = self.classes[emotion_index]
                    confidence = predictions[0][emotion_index] * 100

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"{emotion_label} ({confidence:.2f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
        except Exception as e:
            print(f"Erreur dans recognize_emotions: {str(e)}")  # Debug print

    def start_camera(self, mode):
        try:
            print("Démarrage de la caméra...")  # Debug print
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Impossible d'ouvrir la webcam.")
            self.running = True
            print("Caméra démarrée, lancement update_frame...")  # Debug print
            self.root.after(10, lambda: self.update_frame(mode))
        except Exception as e:
            print(f"Erreur caméra: {str(e)}")  # Debug print
            messagebox.showerror("Erreur", f"Échec de la caméra : {str(e)}")

    def stop_camera(self):
        print("Arrêt de la caméra...")  # Debug print
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def update_frame(self, mode):
        try:
            if self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)

                    if mode == "emotions":
                        self.recognize_emotions(frame)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480))  # Redimensionnement fixe
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.lbl_video.imgtk = imgtk
                    self.lbl_video.configure(image=imgtk)

                    self.root.after(10, lambda: self.update_frame(mode))
                else:
                    print("Erreur de lecture de frame")  # Debug print
                    self.stop_camera()
            else:
                print("Caméra non active")  # Debug print
        except Exception as e:
            print(f"Erreur dans update_frame: {str(e)}")  # Debug print
            self.stop_camera()

    def navigate_to_page(self, page):
        for frame in self.frames.values():
            frame.place_forget()
        self.frames[page].place(relwidth=1, relheight=1)

    def show_help(self):
        messagebox.showinfo(
            "Aide",
            "Cette application permet de reconnaître des émotions en temps réel.\n"
            "Instructions :\n"
            "1. Choisissez 'Emotions Recognition' pour commencer.\n"
            "2. Utilisez 'Stop' pour arrêter le flux vidéo."
        )

    def setup_ui(self):
        self.root.title("Reconnaissance d'Émotions")
        self.root.geometry("800x600")

        try:
            self.bg_image = ImageTk.PhotoImage(Image.open("background.png"))
        except Exception as e:
            print(f"Erreur de chargement du fond: {str(e)}")  # Debug print
            self.bg_image = None

        self.frames = {}
        self.setup_home_page()
        self.setup_mode_selection()
        self.setup_emotions_page()

        self.navigate_to_page("home")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_home_page(self):
        self.frames["home"] = Frame(self.root)
        if self.bg_image:
            Label(self.frames["home"], image=self.bg_image).place(relwidth=1, relheight=1)

        Button(
            self.frames["home"],
            text="Start",
            font=("Helvetica", 18),
            bg="#28a745",
            fg="white",
            command=lambda: self.navigate_to_page("mode_selection")
        ).place(x=325, y=300, width=150, height=50)

        Button(
            self.frames["home"],
            text="Help",
            font=("Helvetica", 14),
            bg="#007bff",
            fg="white",
            command=self.show_help
        ).place(x=325, y=380, width=150, height=40)

    def setup_mode_selection(self):
        self.frames["mode_selection"] = Frame(self.root)
        if self.bg_image:
            Label(self.frames["mode_selection"], image=self.bg_image).place(relwidth=1, relheight=1)

        Button(
            self.frames["mode_selection"],
            text="Emotions Recognition",
            font=("Helvetica", 18),
            bg="#ff5722",
            fg="white",
            command=lambda: self.navigate_to_page("emotions")
        ).place(x=250, y=300, width=300, height=50)

        Button(
            self.frames["mode_selection"],
            text="Back to Home",
            font=("Helvetica", 14),
            bg="#6c757d",
            fg="white",
            command=lambda: self.navigate_to_page("home")
        ).place(x=325, y=460, width=150, height=40)

    def setup_emotions_page(self):
        self.frames["emotions"] = Frame(self.root)
        if self.bg_image:
            Label(self.frames["emotions"], image=self.bg_image).place(relwidth=1, relheight=1)

        # Modification du label vidéo
        self.lbl_video = Label(self.frames["emotions"])
        self.lbl_video.pack(pady=20, expand=True, fill='both')

        control_frame = Frame(self.frames["emotions"])
        control_frame.pack(pady=10)

        Button(
            control_frame,
            text="Start Emotions Recognition",
            font=("Helvetica", 14),
            bg="#28a745",
            fg="white",
            command=lambda: self.start_camera("emotions")
        ).pack(side='left', padx=5)

        Button(
            control_frame,
            text="Stop",
            font=("Helvetica", 14),
            bg="#dc3545",
            fg="white",
            command=self.stop_camera
        ).pack(side='left', padx=5)

        Button(
            control_frame,
            text="Go Back",
            font=("Helvetica", 14),
            bg="#6c757d",
            fg="white",
            command=lambda: self.navigate_to_page("mode_selection")
        ).pack(side='left', padx=5)

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    print("Démarrage de l'application...")  # Debug print
    root = tk.Tk()
    print("Fenêtre Tk créée")  # Debug print
    app = EmotionRecognitionApp(root)
    print("Application initialisée")  # Debug print
    root.mainloop()