import tkinter as tk

root = tk.Tk()
root.geometry("300x150")
root.title("Tes Tkinter")

label = tk.Label(root, text="Tkinker Jalan")
label.pack(pady=40)

root.mainloop()