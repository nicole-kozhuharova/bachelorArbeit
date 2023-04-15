import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Selector")

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.select_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_button.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

if __name__ == '__main__':
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
