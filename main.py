import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt

MOODS = ['Angry', 'Happy', 'Neutral']
GAZES = ['Center', 'Right', 'Left']
GLASSES = ['NoGlasses', 'Glasses']
IMG_SIZE = (128, 128)

# Ensure required directories exist
def ensure_directories():
    for folder in ["dataset", "models", "training_logs"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

class CaptureWindow(tk.Toplevel):
    def __init__(self, parent, on_complete):
        super().__init__(parent)
        self.title("Capture Images for Training")
        self.geometry("700x600")
        self.on_complete = on_complete
        self.cap = cv2.VideoCapture(0)
        self.combinations = [(m, g, gl) for m in MOODS for g in GAZES for gl in GLASSES]
        self.current_idx = 0
        self.captured = 0
        self.max_per_comb = 30
        self.create_widgets()
        self.update_frame()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Follow the prompt and click 'Capture' for each combination.", font=("Arial", 12))
        self.label.pack(pady=10)
        self.prompt = ttk.Label(self, text="", font=("Arial", 14, "bold"))
        self.prompt.pack(pady=10)
        self.canvas = tk.Canvas(self, width=320, height=240)
        self.canvas.pack()
        self.progress = ttk.Progressbar(self, length=400, maximum=len(self.combinations)*self.max_per_comb)
        self.progress.pack(pady=10)
        self.capture_btn = ttk.Button(self, text="Capture", command=self.capture_image)
        self.capture_btn.pack(pady=5)
        self.next_btn = ttk.Button(self, text="Next Combination", command=self.next_combination)
        self.next_btn.pack(pady=5)
        self.finish_btn = ttk.Button(self, text="Finish", command=self.finish, state=tk.DISABLED)
        self.finish_btn.pack(pady=10)
        self.update_prompt()

    def update_prompt(self):
        if self.current_idx < len(self.combinations):
            m, g, gl = self.combinations[self.current_idx]
            self.prompt.config(text=f"Mood: {m}, Gaze: {g}, Glasses: {gl} ({self.captured}/{self.max_per_comb})")
        else:
            self.prompt.config(text="All combinations done!")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((320, 240))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        if self.cap.isOpened():
            self.after(30, self.update_frame)

    def capture_image(self):
        if self.current_idx >= len(self.combinations):
            return
        m, g, gl = self.combinations[self.current_idx]
        folder = os.path.join("dataset", f"{m}_{g}_{gl}")
        os.makedirs(folder, exist_ok=True)
        count = len(os.listdir(folder))
        num_to_capture = self.max_per_comb - self.captured
        for i in range(num_to_capture):
            ret, frame = self.cap.read()
            if not ret:
                continue
            filename = os.path.join(folder, f"img_{count + i + 1}.jpg")
            img = cv2.resize(frame, IMG_SIZE)
            cv2.imwrite(filename, img)
            self.captured += 1
            self.progress['value'] += 1
            self.update_prompt()
        if self.captured >= self.max_per_comb:
            self.capture_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL)

    def next_combination(self):
        self.current_idx += 1
        self.captured = 0
        self.capture_btn.config(state=tk.NORMAL)
        self.next_btn.config(state=tk.DISABLED)
        if self.current_idx >= len(self.combinations):
            self.capture_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.finish_btn.config(state=tk.NORMAL)
        self.update_prompt()

    def finish(self):
        self.cap.release()
        self.destroy()
        self.on_complete()

class TrainWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Train Model")
        self.geometry("500x400")
        self.create_widgets()
        self.model = None

    def create_widgets(self):
        self.label = ttk.Label(self, text="Training Model...", font=("Arial", 14))
        self.label.pack(pady=10)
        self.progress = ttk.Progressbar(self, length=400, mode='determinate', maximum=10)
        self.progress.pack(pady=10)
        self.timer_label = ttk.Label(self, text="Elapsed Time: 0.0s", font=("Arial", 10))
        self.timer_label.pack(pady=5)
        self.start_time = datetime.now()
        self.after(100, self.train_model)

    def train_model(self):
        X, y = [], []
        for idx, (m, g, gl) in enumerate([(m, g, gl) for m in MOODS for g in GAZES for gl in GLASSES]):
            folder = os.path.join("dataset", f"{m}_{g}_{gl}")
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    X.append(img)
                    y.append(idx)
        X = np.array(X, dtype=np.float32) / 255.0
        y = to_categorical(y, num_classes=18)
        self.model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(18, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('training_logs', f'model_{now}')
        os.makedirs(log_dir, exist_ok=True)
        csv_logger = CSVLogger(os.path.join(log_dir, 'metrics.csv'))
        import time
        start = time.time()
        history = self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[csv_logger], verbose=0)
        end = time.time()
        model_path = os.path.join('models', f'model_{now}.h5')
        self.model.save(model_path)
        self.progress['value'] = 10
        self.label.config(text=f"Training Complete! Model saved as {model_path}")
        self.plot_history(history, log_dir)
        elapsed = end - start
        self.timer_label.config(text=f"Elapsed Time: {elapsed:.1f}s")
        messagebox.showinfo("Training Complete", f"Model saved as {model_path}")
        self.destroy()

    def plot_history(self, history, log_dir):
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title('Accuracy')
        plt.savefig(os.path.join(log_dir, 'accuracy.png'))
        plt.close()
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(log_dir, 'loss.png'))
        plt.close()

class TestWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Test Model")
        self.geometry("600x500")
        self.model = None
        self.create_widgets()

    def create_widgets(self):
        self.model_label = ttk.Label(self, text="Select Model:")
        self.model_label.pack(pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(pady=5)
        self.model_combo['values'] = self.get_models()
        if self.model_combo['values']:
            self.model_combo.current(0)
        self.load_btn = ttk.Button(self, text="Load Model", command=self.load_model)
        self.load_btn.pack(pady=5)
        self.test_type = tk.StringVar(value="Webcam")
        self.webcam_radio = ttk.Radiobutton(self, text="Webcam", variable=self.test_type, value="Webcam")
        self.webcam_radio.pack()
        self.image_radio = ttk.Radiobutton(self, text="Image", variable=self.test_type, value="Image")
        self.image_radio.pack()
        self.start_btn = ttk.Button(self, text="Start Test", command=self.start_test)
        self.start_btn.pack(pady=10)
        self.canvas = tk.Canvas(self, width=320, height=240)
        self.canvas.pack(pady=10)
        self.result_label = ttk.Label(self, text="Result: ", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def get_models(self):
        models = []
        if os.path.exists('models'):
            for fname in os.listdir('models'):
                if fname.endswith('.h5'):
                    models.append(fname)
        return models

    def load_model(self):
        model_name = self.model_var.get()
        if model_name:
            self.model = load_model(os.path.join('models', model_name))
            messagebox.showinfo("Model Loaded", f"Loaded {model_name}")
            # Prevent reopening home window by disabling parent
            if hasattr(self.master, 'withdraw'):
                self.master.withdraw()

    def start_test(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return
        if self.test_type.get() == "Webcam":
            self.test_webcam()
        else:
            self.test_image()

    def test_webcam(self):
        # Improved webcam capture with instructions and status
        import threading
        def webcam_thread():
            status_popup = tk.Toplevel(self)
            status_popup.title("Camera Status")
            status_label = ttk.Label(status_popup, text="Opening camera, please wait...")
            status_label.pack(padx=20, pady=20)
            cap = cv2.VideoCapture(0)
            opened = cap.isOpened()
            status_label.config(text="Camera opened! Press 'c' to capture, 'q' to quit.")
            self.after(1000, status_popup.destroy)
            while opened:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Webcam - Press c to capture, q to quit', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    img = cv2.resize(frame, IMG_SIZE)
                    img_arr = img.astype(np.float32) / 255.0
                    img_arr = np.expand_dims(img_arr, axis=0)
                    pred = self.model.predict(img_arr)[0]
                    idx = np.argmax(pred)
                    m, g, gl = self.decode_idx(idx)
                    self.result_label.config(text=f"Result: Mood={m}, Gaze={g}, Glasses={gl}")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_disp = Image.fromarray(rgb)
                    img_disp = img_disp.resize((320, 240))
                    imgtk = ImageTk.PhotoImage(image=img_disp)
                    self.canvas.imgtk = imgtk
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                elif key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        threading.Thread(target=webcam_thread).start()

    def test_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Invalid image file.")
            return
        img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_disp = Image.fromarray(img_disp)
        img_disp = img_disp.resize((320, 240))
        imgtk = ImageTk.PhotoImage(image=img_disp)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        # Ensure correct preprocessing: resize, 3 channels, normalize
        img_arr = cv2.resize(img, IMG_SIZE)
        if img_arr.shape[-1] != 3:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        img_arr = img_arr.astype(np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        pred = self.model.predict(img_arr)[0]
        idx = np.argmax(pred)
        m, g, gl = self.decode_idx(idx)
        self.result_label.config(text=f"Result: Mood={m}, Gaze={g}, Glasses={gl}")

    def decode_idx(self, idx):
        m = MOODS[idx // 6]
        g = GAZES[(idx % 6) // 2]
        gl = GLASSES[idx % 2]
        return m, g, gl

class MoodGazeGlassesApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mood, Gaze & Glasses Detection")
        self.geometry("500x350")
        self.resizable(False, False)
        self.create_widgets()

    def create_widgets(self):
        title = ttk.Label(self, text="Mood, Gaze & Glasses Detection", font=("Arial", 18, "bold"))
        title.pack(pady=20)
        train_btn = ttk.Button(self, text="Capture Images & Train New Model", command=self.train_new_model)
        train_btn.pack(pady=10, ipadx=10, ipady=5)
        test_btn = ttk.Button(self, text="Test Model", command=self.test_model)
        test_btn.pack(pady=10, ipadx=10, ipady=5)
        clear_btn = ttk.Button(self, text="Clear All (Delete Models & Dataset)", command=self.clear_all)
        clear_btn.pack(pady=10, ipadx=10, ipady=5)
        quit_btn = ttk.Button(self, text="Quit", command=self.quit)
        quit_btn.pack(pady=10, ipadx=10, ipady=5)

    def train_new_model(self):
        def after_capture():
            TrainWindow(self)
        CaptureWindow(self, on_complete=after_capture)

    def test_model(self):
        TestWindow(self)

    def clear_all(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to delete ALL models and dataset images? This cannot be undone."):
            # Delete all models
            models_dir = os.path.join("models")
            if os.path.exists(models_dir):
                for fname in os.listdir(models_dir):
                    fpath = os.path.join(models_dir, fname)
                    if os.path.isfile(fpath):
                        try:
                            os.remove(fpath)
                        except Exception as e:
                            print(f"Error deleting {fpath}: {e}")
            # Delete all images in dataset subfolders
            for m in MOODS:
                for g in GAZES:
                    for gl in GLASSES:
                        folder = os.path.join("dataset", f"{m}_{g}_{gl}")
                        if os.path.exists(folder):
                            for fname in os.listdir(folder):
                                fpath = os.path.join(folder, fname)
                                if os.path.isfile(fpath):
                                    try:
                                        os.remove(fpath)
                                    except Exception as e:
                                        print(f"Error deleting {fpath}: {e}")
            messagebox.showinfo("Cleared", "All models and dataset images have been deleted.")

if __name__ == "__main__":
    ensure_directories()
    app = MoodGazeGlassesApp()
    app.mainloop()