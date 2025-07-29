# 🚀 AI-Powered Salary Predictor

## 📋 Project Overview

**Objective**: Develop a machine learning model to predict whether an employee earns >$50K or ≤$50K based on demographic and work-related features.

**Dataset**: Adult Census Income Dataset (~48K records, 14 features)

**Live Demo**: Professional Streamlit web application with dark theme UI

## 🎯 Key Features

- ✅ **86.32% Accuracy** Gradient Boosting model
- ✅ **Dark Theme** professional web interface
- ✅ **Real-time predictions** with confidence scores
- ✅ **Batch processing** for CSV file uploads
- ✅ **Interactive visualizations** and insights
- ✅ **Mobile-responsive** design

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Salary-Predictor.git
cd Salary-Predictor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model (optional - pre-trained model included)**

```bash
python improved_salary_model.py
```

4. **Run the web application**

```bash
streamlit run app.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Model Performance

## 📊 Model Performance

### Algorithm Comparison

| Algorithm             | Accuracy   | Notes                |
| --------------------- | ---------- | -------------------- |
| **Gradient Boosting** | **86.32%** | 🏆 **Best Model**    |
| Random Forest         | 85.57%     | Strong ensemble      |
| SVM                   | 84.72%     | Good performance     |
| Logistic Regression   | 82.62%     | Fast & interpretable |
| KNN                   | 82.89%     | Distance-based       |
| Decision Tree         | 81.08%     | Simple baseline      |

### Detailed Metrics

- **Accuracy**: 86.32%
- **Precision**: 88% (≤50K), 79% (>50K)
- **Recall**: 95% (≤50K), 61% (>50K)
- **F1-Score**: 91% (≤50K), 69% (>50K)

## 🏗️ Technical Architecture

### Data Pipeline

```
Raw Data → Cleaning → Feature Encoding → Scaling → Model Training → Evaluation
```

### Features (13 input variables)

1. **Demographics**: Age, Gender, Race, Native Country
2. **Education**: Educational Number (5-16 scale)
3. **Work**: Work Class, Occupation, Hours per Week
4. **Personal**: Marital Status, Relationship
5. **Financial**: Capital Gain, Capital Loss

## 💡 Key Insights

### High Earning Predictors

- 🎓 Education ≥ Bachelor's degree
- 👔 Executive/Professional roles
- ⏰ Full-time work (40+ hours)
- 💍 Married status
- 🎂 Age: 35-55 years

## 📁 Project Structure

```
Salary-Predictor/
│
├── app.py                          # Streamlit web application
├── improved_salary_model.py        # ML pipeline & training
├── improved_salary_model.pkl       # Trained model file
├── adult 3.csv                     # Dataset
├── Figure_1.png                    # Performance visualization
├── requirements.txt                # Dependencies
├── PROJECT_REPORT.md               # Detailed report
└── README.md                       # This file
```

## 🎨 Screenshots

### Dark Theme Interface

- Professional dark theme with cyan accents
- Centered input form with organized sections
- Real-time predictions with confidence scores
- Batch processing capabilities

## 🔧 Technologies Used

- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Data Processing**: Scipy

## 📈 Use Cases

1. **HR Analytics**: Salary benchmarking and compensation planning
2. **Recruitment**: Salary offer optimization
3. **Policy Making**: Income inequality analysis
4. **Career Guidance**: Education and career path recommendations

## 🚀 Future Enhancements

- [ ] Hyperparameter tuning with GridSearch
- [ ] Deep learning models (Neural Networks)
- [ ] Feature importance analysis
- [ ] User authentication
- [ ] Prediction history
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Docker containerization
- [ ] REST API endpoints

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Adult Census Income Dataset from UCI ML Repository
- Streamlit community for excellent documentation
- Scikit-learn team for robust ML library

---

⭐ **Star this repository if you found it helpful!**
