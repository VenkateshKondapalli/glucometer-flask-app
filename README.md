# 🩸 Glucometer Glucose Forecaster  
**A Flask web application powered by a Transformer-based deep learning model for real-time blood glucose forecasting.**

---

## 🚀 Overview  
**Glucometer Glucose Forecaster** predicts a patient’s blood glucose levels **30 minutes into the future** using historical data from XML files.  
The app features an intuitive web interface, interactive visualizations, and a powerful Transformer model optimized for time-series prediction.

---

## ✨ Features  

- 🕒 **30-Minute Glucose Forecasting** — Predict upcoming glucose levels based on past readings.  
- 📂 **XML File Upload** — Upload patient data in a structured XML format.  
- 📊 **Interactive Visualization** — Explore real-time glucose trends and forecasts via **Chart.js** graphs.  
- 🤖 **Deep Learning Model (Transformer)** — Built with **TensorFlow** and **Keras** for accurate time-series prediction.  
- 🧩 **Modular Flask Architecture** — Scalable **Application Factory** structure with Blueprints for clean project organization.  

---

## 🧠 Tech Stack  

| Category | Technologies |
|-----------|--------------|
| **Backend** | Flask, Python |
| **Machine Learning** | TensorFlow, Keras, Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Frontend** | HTML, JavaScript, Chart.js |

---

## 🏗️ Project Structure  

```
/glucometer-flask-app/
├── run.py                  # Entry point to start the Flask app
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
└── glucometer_app/         # Main application package
    ├── __init__.py         # Flask Application Factory
    ├── ml_utils.py         # Model loading & prediction logic
    ├── static/             # CSS and JS files
    ├── templates/          # HTML templates
    └── main/               # Application blueprint
        ├── __init__.py
        └── routes.py       # App routes and logic
```

---

## ⚙️ Setup and Installation  

### 1️⃣ Clone the Repository  
```bash
git clone <your-repository-url>
cd glucometer-flask-app
```

### 2️⃣ Create and Activate a Virtual Environment  
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application  
```bash
python run.py
```

Then open your browser and navigate to:  
👉 **http://127.0.0.1:5000**

---

## 📈 Example Output  
After uploading your XML file, you’ll see:  
- A **graph** displaying historical glucose data.  
- The **forecasted glucose values** for the next 30 minutes overlaid on the same chart.  

---

## 📬 Future Enhancements  
- Add patient authentication and profile management.  
- Integrate real-time glucose data from IoT-based glucometers.  
- Deploy the app using Docker or AWS.  

---

## 🧑‍💻 Author  
**Venkatesh Kondapalli**  
📧 https://www.linkedin.com/in/venkatesh-kondapalli 
💻 Passionate about AI, ML, and Intelligent Healthcare Systems  
