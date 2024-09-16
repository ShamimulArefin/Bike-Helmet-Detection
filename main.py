import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from PIL import Image, ImageTk

# video logic
model = YOLO('best.pt')
def select_video(right_frame):
    # Define allowed video file extensions
    allowed_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv']

    # Prompt user to select a video file
    video_path = filedialog.askopenfilename()

    if video_path:
        # Check if the selected file has an allowed extension
        if any(video_path.lower().endswith(ext) for ext in allowed_extensions):
            cap = cv2.VideoCapture(video_path)
            process_video(cap, right_frame)
        else:
            # Display an error message if the selected file is not a valid video file
            messagebox.showerror("Invalid File Type", "Please select a valid video file.")

# def select_video(right_frame):
#     video_path = filedialog.askopenfilename()
#     if video_path:
#         cap = cv2.VideoCapture(video_path)
#         process_video(cap, right_frame)

def access_camera(right_frame):
    cap = cv2.VideoCapture(0)  # Accessing live camera feed
    process_video(cap, right_frame)

def process_video(cap, right_frame):
    with open("COCO.txt", "r") as my_file:
        data = my_file.read()
    class_list = data.split("\n")
    count = 0

    def quit_video():
        nonlocal quit
        quit = True

    quit = False

    while True:
        ret, frame = cap.read()
        if not ret or quit:
            break
        count += 1

        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]

            if c == 'helmet':
                # Save frame with helmet detected
                frame_name = f"helmet_frame_{count}.jpg"
                cv2.imwrite(frame_name, frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        # Convert frame to ImageTk format for Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img)

        # Display the frame in the right frame
        right_frame_label = ttk.Label(right_frame, image=img_tk)
        right_frame_label.image = img_tk  # Keep a reference to prevent garbage collection
        right_frame_label.grid(row=0, column=0, padx=10, pady=10)

        # Add buttons for controlling video playback
        stop_button = ttk.Button(right_frame, text="Quit", command=quit_video)
        stop_button.grid(row=1, column=0, padx=10, pady=5)

        right_frame.update_idletasks()
        right_frame.update()

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if quit:
        right_frame_label.grid_forget()
        # Display message in the right frame
        right_message = ttk.Label(right_frame, text="Please select any action from left area")
        right_message.grid(row=0, column=0, padx=10, pady=10)

# video logic end

def create_home_frame(root):
    home_frame = ttk.Frame(root, padding="10 10 10 10")
    home_frame.grid(row=0, column=0, sticky="nsew")

    # Configure the grid layout for the home frame
    home_frame.rowconfigure([0, 1, 2], weight=1)
    home_frame.columnconfigure(0, weight=1)

    # Header
    header = ttk.Label(home_frame, text="Automatic Helmet Detection", font=("Helvetica", 26))
    header.grid(row=0, column=0, pady=20)

    # Body with a paragraph and button
    body = ttk.Frame(home_frame)
    body.grid(row=1, column=0, pady=20)
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)

    paragraph = ttk.Label(body, text="Detect if bike riders are wearing helmets with ease. Click below to get started.", wraplength=600, justify="center", font=("Helvetica", 18))
    paragraph.pack(pady=10)

    # Create a custom style for the button
    style = ttk.Style()
    style.configure("Custom.TButton", padding=10, font=("Helvetica", 16), background="#5352ed", borderwidth=0, relief="flat")

    get_started_button = ttk.Button(body, text="Get Started", style='Custom.TButton', command=lambda: create_main_frame(root))
    get_started_button.pack(pady=20, ipadx=20, ipady=10, anchor="center")

    # Footer
    footer = ttk.Label(home_frame, text="© 2024 Automatic Helmet Detection by Arefin", font=("Helvetica", 12))
    footer.grid(row=2, column=0, pady=20)

    return home_frame

def show_info_window():
    # Create a new window
    info_window = tk.Toplevel()
    info_window.title("Application Information")

    # Heading
    heading_label = ttk.Label(info_window, text="Automatic Helmet Detection for Bike Riders using Machine Learning", font=("Helvetica", 16))
    heading_label.pack(pady=10)

    # Paragraph
    paragraph_text = ("This project is designed to detect whether bike riders are wearing helmets. "
                      "It aims to improve safety and compliance on the roads.")
    paragraph_label = ttk.Label(info_window, text=paragraph_text, wraplength=300, justify="left")
    paragraph_label.pack(pady=10)

    # Developer Information
    dev_info_text = "This project is developed by Arefin, Software Engineering, Zhengzhou University"
    dev_info_label = ttk.Label(info_window, text=dev_info_text, wraplength=300, justify="left")
    dev_info_label.pack(pady=10)

# main frame start

def create_main_frame(root):
    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky="nsew")

    # Configure the grid layout
    main_frame.columnconfigure(0, weight=1, uniform="uniform")
    main_frame.columnconfigure(1, weight=4, uniform="uniform")
    main_frame.rowconfigure(0, weight=1)

    # Create left frame
    left_frame = ttk.Frame(main_frame, padding="10 10 10 10", relief="sunken")
    left_frame.grid(row=0, column=0, sticky="nsew")
    left_frame.rowconfigure([0, 1, 2], weight=1)
    left_frame.columnconfigure(0, weight=1)

    # Create right frame
    right_frame = ttk.Frame(main_frame, padding="10 10 10 10", relief="sunken")
    right_frame.grid(row=0, column=1, sticky="nsew")
    right_frame.rowconfigure(0, weight=1)
    right_frame.columnconfigure(0, weight=1)

    # add content to right frame
    # Display message in the right frame
    right_message = ttk.Label(right_frame, text="Please select any action from left area")
    right_message.grid(row=0, column=0, padx=10, pady=10)

    # Add content to the left frame

    # Heading section
    left_heading = ttk.Label(left_frame, text="Select Actions", font=("Helvetica", 16))
    left_heading.grid(row=0, column=0, padx=10, pady=10)

    # Button section
    button_frame = ttk.Frame(left_frame)
    button_frame.grid(row=1, column=0, pady=20, sticky="n")
    button_frame.columnconfigure(0, weight=1)
    
    real_time_button = ttk.Button(button_frame, text="Real-time Camera", command=lambda: access_camera(right_frame))
    real_time_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

    upload_video_button = ttk.Button(button_frame, text="Upload Video", command=lambda: select_video(right_frame))
    upload_video_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    # About section
    about_frame = ttk.Frame(left_frame)
    about_frame.grid(row=2, column=0, pady=20, sticky="s")
    about_frame.columnconfigure(0, weight=1)
    
    about_paragraph = ttk.Label(about_frame, text="This app detects whether bike riders are wearing helmets. It's designed to improve safety and compliance.", wraplength=150, justify="left")
    about_paragraph.grid(row=0, column=0, padx=10, pady=10)

    # Info button
    info_button = ttk.Button(about_frame, text="Info", command=show_info_window)
    info_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    return main_frame, right_frame


# main frame end

def show_frame(frame):
    frame.tkraise()

def create_gui():
    root = tk.Tk()
    root.title("Automatic Helmet Detection")
    root.geometry("1020x600")

    # Configure root window's grid
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Create all frames
    home_frame = create_home_frame(root)
    main_frame, right_frame = create_main_frame(root)

    for frame in (home_frame, main_frame):
        frame.grid(row=0, column=0, sticky="nsew")

    show_frame(home_frame)  # Show home frame first

    root.mainloop()

if __name__ == "__main__":
    create_gui()