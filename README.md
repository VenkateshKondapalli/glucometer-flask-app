# 🩸 Glucometer Glucose Forecaster

**A Flask web application powered by a Transformer-based deep learning model for real-time blood glucose forecasting.**

---

## 🚀 Overview

**Glucometer Glucose Forecaster** predicts a patient's blood glucose levels **30 minutes into the future** using historical data from XML files. The app features an intuitive web interface, interactive visualizations, and a powerful Transformer model optimized for time-series prediction.

### Why Use This?
- 🏥 Anticipate glucose fluctuations before they happen
- 📊 Make data-driven healthcare decisions
- 🎯 Easy-to-use interface with drag-and-drop functionality
- 🤖 Powered by state-of-the-art AI technology

---

## ✨ Features

- 🕒 **30-Minute Glucose Forecasting** — Predict upcoming glucose levels based on past readings
- 📂 **XML File Upload** — Upload patient data in a structured XML format with drag-and-drop support
- 📊 **Interactive Visualization** — Explore real-time glucose trends and forecasts via Chart.js graphs
- 🤖 **Deep Learning Model (Transformer)** — Built with TensorFlow and Keras for accurate time-series prediction
- 🧩 **Modular Flask Architecture** — Scalable Application Factory structure with Blueprints
- 📈 **Statistics Dashboard** — View current, predicted, and trend values at a glance
- 💾 **Download Charts** — Save prediction graphs as PNG images

---

## 🧠 Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Backend** | Flask, Python 3.8+ |
| **Machine Learning** | TensorFlow, Keras, Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Visualization** | Chart.js |

---

## 🏗️ Project Structure

```
glucometer-flask-app/
├── run.py                  # Entry point to start the Flask app
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
└── glucometer_app/         # Main application package
    ├── __init__.py         # Flask Application Factory
    ├── ml_utils.py         # Model loading & prediction logic
    ├── static/             # CSS and JS files
    ├── templates/          # HTML templates
    │   ├── index.html      # Upload page
    │   └── result.html     # Results page
    └── main/               # Application blueprint
        ├── __init__.py
        └── routes.py       # App routes and logic
```

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/VenkateshKondapalli/glucometer-flask-app.git
cd glucometer-flask-app
```

**2. Create and Activate Virtual Environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Application**
```bash
python run.py
```

**5. Open in Browser**
```
http://127.0.0.1:5000
```

---

## 🎯 Usage

1. **Upload XML File** — Drag and drop or click to select your patient's glucose data XML file
2. **Submit** — Click "Get Prediction" button
3. **View Results** — See the prediction graph with statistics
4. **Download** — Save the chart as an image for your records

### XML File Format Example
```xml
<glucose_data>
  <reading>
    <timestamp>2025-10-14 10:00:00</timestamp>
    <value>120</value>
  </reading>
  <!-- More readings... -->
</glucose_data>
```

---

## 📈 What You'll See

After uploading, the results page displays:

**Statistics Cards:**
- Current Glucose Level (mg/dL)
- Predicted Level (30 minutes ahead)
- Trend (increase/decrease)

**Interactive Chart:**
- Blue line: Your historical glucose data
- Purple dashed line: 30-minute forecast
- Green line: Normal glucose range reference

---

## 🔒 Important Notes

⚠️ **Medical Disclaimer:**
- This application is for **informational and educational purposes only**
- Predictions should **NOT replace professional medical advice**
- Always consult with healthcare professionals for treatment decisions
- Patient data is processed temporarily and not stored

---

## 🚧 Future Enhancements

- [ ] User authentication and patient profiles
- [ ] Real-time data integration from IoT glucometers
- [ ] Mobile app development
- [ ] API for third-party integrations
- [ ] Advanced analytics and pattern recognition
- [ ] Cloud deployment (AWS/Heroku/Docker)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

**Steps to Contribute:**
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Venkatesh Kondapalli**

🔗 [LinkedIn](https://www.linkedin.com/in/venkatesh-kondapalli)  
💻 Passionate about AI, ML, and Intelligent Healthcare Systems

---

## 🌟 Support

If you find this project helpful, please give it a ⭐️ on GitHub!

For questions or support:
- Open an issue on GitHub
- Connect on LinkedIn

---

**Made with ❤️ for Healthcare Innovation**