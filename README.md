# 🚗 CO2 Emissions Predictor

**CO2 Emissions Predictor** is a machine learning application that uses a **Simple Linear Regression (SLR)** model to predict the carbon dioxide emissions (g/km) of a vehicle based on its engine size (L). This project demonstrates how to build, evaluate, and deploy a regression model using Python and Streamlit.

![CO2 Emissions Predictor Banner](banner_md.jpeg)

---

## 🌟 Features
- **Simple Linear Regression (SLR) Model**: Efficient and accurate prediction of CO2 emissions based on engine size.
- **Interactive User Interface**: Built using Streamlit for an intuitive experience.
- **Real-Time Predictions**: Adjust engine size dynamically to get instant CO2 emission predictions.
- **Data Visualization**: Displays scatterplots of sample data with the regression line for better understanding.
- **Eco-Friendly Insights**: Learn about the relationship between engine size and emissions.

---

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **Streamlit**: Framework for creating the interactive web application.
- **Scikit-learn**: Machine learning library used for training the SLR model.
- **Matplotlib**: For data visualization.

---

## 🚀 How It Works
1. **Input Engine Size**: Use the slider to select the engine size (in liters).
2. **Real-Time Prediction**: The app instantly predicts the CO2 emissions based on the input.
3. **Visualize Data**: See a scatterplot of engine size vs. CO2 emissions along with the regression line.

---

## 🖥️ Installation and Usage

### Prerequisites
- Python 3.8 or higher installed on your machine.

### Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/ashishpatel8736/CO2-Emissions-Predictor.git
   cd CO2-Emissions-Predictor
```

### Step 2: Install Dependencies
Ensure you have Python installed. Run the following to install the required libraries:
```bash
pip install -r requirements.txt

```

### Step 3: Start the Application
Run the Streamlit app:
```bash
streamlit run app.py
```

### Step 4: Open your browser and go to:

```bash
http://localhost:8501

```

## 📂 Repository Structure
```plaintext
CO2-Emissions-Predictor-using-Streamlit/

├── app.py                     Streamlit app file
├── slr_model.pkl              Pre-trained Simple Linear Regression model
├── README.md                  Project documentation
├── requirements.txt           Python dependencies
├── LICENSE                    Licence file
├── banner_md.jpeg             banner image
├── icons8-github-50.png
```

---

## 📊 Sample Data
Here is an example of the dataset used for training the SLR model:

| Engine Size (L)  | CO2 Emissions (g/km) |
|------------------|----------------------|
| 1.5              | 145                  |
| 2.0              | 185                  |
| 3.0              | 250                  |
| 4.0              | 320                  |
| 5.0              | 400                  |


---

## 🎯 Future Enhancements
- Implement a **Multiple Linear Regression (MLR)** model to include additional features like fuel consumption.
- Add support for uploading custom datasets.
- Provide downloadable results and summary reports.




---


## 🤝 Contributing
Contributions are welcome! If you'd like to contribute, please:

1. **Fork the repository**.
2. **Create a feature branch**.
3. **Submit a pull request**.



## 🙌 Acknowledgements
- **Scikit-learn** for providing robust machine learning tools.
- **Streamlit** for enabling easy deployment of ML apps.

---
## 🛡️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author  
**Ashish Patel**  
[![GitHub](icons8-github-50.png)](https://github.com/ashishpatel8736) | [![LinkedIn](https://img.icons8.com/ios-filled/50/0077b5/linkedin.png)](https://www.linkedin.com/in/ashishpatel8736)


