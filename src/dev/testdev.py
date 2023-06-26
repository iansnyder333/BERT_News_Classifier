import tkinter as tk

window = tk.Tk()
label = tk.Label(text="Hello, Tkinter", fg="white", bg="black", width=10, height=10)
button = tk.Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
)
label.pack()
button.pack()
entry = tk.Entry(fg="yellow", bg="blue", width=50)
entry.pack()
window.mainloop()
