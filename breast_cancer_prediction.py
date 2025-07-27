import customtkinter as ctk
import pickle
import numpy as np
from tkinter import messagebox

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

with open('best_lr_model_7features.pkl', 'rb') as f:
    model=pickle.load(f)

with open('scaler_7features.pkl', 'rb') as f:
    scaler=pickle.load(f)

features=[
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity"
]

app=ctk.CTk()
app.title("Breast Cancer Prediction")
app.geometry("700x750")

app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)

entries={}

header_label=ctk.CTkLabel(
    app,
    text="Please Enter Your Values",
    font=("Arial", 24, "bold"),
    width=500
)
header_label.grid(row=0, column=0, columnspan=2, pady=(20, 40), sticky="ew", ipady=35)
app.grid_columnconfigure(0, minsize=100)

for i, feature in enumerate(features, start=1):
    label=ctk.CTkLabel(
        app,
        text=feature.title(),
        anchor="center",
        justify="center"
    )
    label.grid(row=i, column=0, padx=10, pady=10, sticky="ew")

    entry=ctk.CTkEntry(app, width=250, justify="center")
    entry.grid(row=i, column=1, padx=10, pady=10)

    entries[feature]=entry

def predict():
    try:
        input_data=[]

        for feature in features:
            val=entries[feature].get()
            if val.strip()=="":
                raise ValueError(f"'{feature}' field cannot be empty.")
            input_data.append(float(val)) 
        
        input_array=np.array(input_data).reshape(1, -1)
        input_scaled=scaler.transform(input_array)

        prediction=model.predict(input_scaled)[0]
        proba=model.predict_proba(input_scaled)[0]

        if prediction == 0:
            diagnosis = "⚠️ Malignant"
            suggestion = "Please consult your doctor immediately. Early diagnosis saves lives."
            color = "red"
        else:
            diagnosis = "✅ Benign"
            suggestion = "Everything looks fine. Regular checkups are still important."
            color = "green"

        result_window = ctk.CTkToplevel(app)
        result_window.title("Result")
        result_window.geometry("450x250")

        result_window.update_idletasks()
        w = result_window.winfo_screenwidth()
        h = result_window.winfo_screenheight()
        size = tuple(int(x) for x in result_window.geometry().split('+')[0].split('x'))
        x = w // 2 - size[0] // 2
        y = h // 2 - size[1] // 2
        result_window.geometry(f"{size[0]}x{size[1]}+{x}+{y}")

        label_result = ctk.CTkLabel(
            result_window,
            text=diagnosis,
            text_color=color,
            font=("Arial", 22)
        )
        label_result.pack(pady=40)

        label_prob = ctk.CTkLabel(
            result_window,
            text=f"Probability: {proba[prediction]*100:.2f}%",
            font=("Arial", 16)
        )
        label_prob.pack(pady=10)

        label_suggestion = ctk.CTkLabel(
            result_window,
            text=suggestion,
            wraplength=400,
            justify="center",
            font=("Arial", 14)
        )
        label_suggestion.pack(pady=20)

    except ValueError as ve:
       
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        
        messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

predict_button = ctk.CTkButton(
    app, text="📊 Predict",
    command=predict,
    width=180,
    height=50,
    font=("Arial", 16)
)
predict_button.grid(row=len(features)+1, column=0, columnspan=2, pady=30)

app.mainloop()