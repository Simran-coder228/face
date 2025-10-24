import cv2
import threading
from deepface import DeepFace
from PIL import Image, ImageTk
import tkinter as tk

# --- GLOBALS ---
running = False
cap = None

# --- FUNCTIONS ---
def start_camera():
    global running, cap
    running = True
    cap = cv2.VideoCapture(0)
    show_frame()

def stop_camera():
    global running, cap
    running = False
    if cap:
        cap.release()
    lbl_video.config(image='')

def show_frame():
    global cap, running
    if running:
        ret, frame = cap.read()
        if ret:
            # Analyze emotion in a separate thread (to keep UI smooth)
            threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True).start()

            # Convert frame for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
        lbl_video.after(10, show_frame)

def analyze_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        lbl_status.config(text=f"Detected Emotion: {emotion.upper()}")
    except Exception:
        lbl_status.config(text="No face detected")

# --- GUI SETUP ---
root = tk.Tk()
root.title("Emotion Detection App")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

# Center all content
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

main_frame = tk.Frame(root, bg="#1e1e1e")
main_frame.grid(sticky="nsew")
main_frame.columnconfigure(0, weight=1)
for i in range(5):
    main_frame.rowconfigure(i, weight=1)

lbl_title = tk.Label(main_frame, text="😊 Real-Time Emotion Detector 😊", 
                     font=("Arial", 22, "bold"), bg="#1e1e1e", fg="white")
lbl_title.grid(row=0, column=0, pady=(20, 10))

lbl_video = tk.Label(main_frame, bg="#1e1e1e")
lbl_video.grid(row=1, column=0, pady=10)

lbl_status = tk.Label(main_frame, text="Click 'Start' to begin", 
                      font=("Arial", 16), bg="#1e1e1e", fg="#00ffcc")
lbl_status.grid(row=2, column=0, pady=10)

frame_btns = tk.Frame(main_frame, bg="#1e1e1e")
frame_btns.grid(row=3, column=0, pady=20)

btn_start = tk.Button(frame_btns, text="Start Camera", command=start_camera, 
                      font=("Arial", 14, "bold"), bg="#00cc88", fg="white", width=14)
btn_start.grid(row=0, column=0, padx=15)

btn_stop = tk.Button(frame_btns, text="Stop Camera", command=stop_camera, 
                     font=("Arial", 14, "bold"), bg="#ff4444", fg="white", width=14)
btn_stop.grid(row=0, column=1, padx=15)

root.mainloop()