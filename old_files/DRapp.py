import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load your trained model 
model = tf.keras.models.load_model('./trained_model/DR3.keras')

# Set up the GUI
class DRapp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        # Canvas size (scaled up for easier drawing)
        self.canvas_width = 280
        self.canvas_height = 280
        self.scale = 10  # 28x28 -> 280x280

        # Create canvas
        self.canvas = Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white', cursor='cross')
        self.canvas.grid(row=0, column=0, columnspan=3, pady=10)

        self.draw_grid()

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Draw object to capture the image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)  # Grayscale, black background
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        self.btn_predict = Button(root, text="Predict", command=self.predict_digit, bg="lightblue", font=("Arial", 12))
        self.btn_predict.grid(row=1, column=0, padx=5, pady=10)

        self.btn_clear = Button(root, text="Clear", command=self.clear_canvas, bg="lightcoral", font=("Arial", 12))
        self.btn_clear.grid(row=1, column=1, padx=5, pady=10)

        self.btn_quit = Button(root, text="Quit", command=root.quit, bg="lightgray", font=("Arial", 12))
        self.btn_quit.grid(row=1, column=2, padx=5, pady=10)

        # Label for prediction
        self.prediction_label = Label(root, text="Draw a digit and click Predict", font=("Arial", 14), fg="gray")
        self.prediction_label.grid(row=2, column=0, columnspan=3, pady=10)
    
    def draw_grid(self):
        """Draw a grid with spacing equal to scale (10px), representing the 28x28 pixel grid."""
        for x in range(0, self.canvas_width, self.scale):
            self.canvas.create_line(x, 0, x, self.canvas_height, fill='lightgray', tags="grid")
        for y in range(0, self.canvas_height, self.scale):
            self.canvas.create_line(0, y, self.canvas_width, y, fill='lightgray', tags="grid")

    def paint(self, event):
        r = 10  # Brush radius
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='white')  # PIL uses white for digits

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Canvas cleared", fg="gray")

    def predict_digit(self):
        # Resize image to 28x28 and preprocess
        img = self.image.resize((28, 28), Image.LANCZOS)
        img = ImageOps.invert(img)  # Invert: model trained on black digits on white background
        img = np.array(img)
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 784)  # Flatten

        # Predict
        prediction = model.predict(img, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Update label
        self.prediction_label.config(
            text=f"Predicted Digit: {digit} (Confidence: {confidence:.2f})",
            fg="green" if confidence > 0.5 else "red"
        )

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DRapp(root)
    root.mainloop()