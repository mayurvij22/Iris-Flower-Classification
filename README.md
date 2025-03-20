🌸 Iris Flower Classification
📌 Project Overview
This project implements a machine learning model to classify Iris flowers into three species:

Setosa
Versicolor
Virginica
The dataset used is the famous Iris dataset, which consists of 150 samples with four features:

Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)
The goal is to train a model that can accurately classify an Iris flower species based on these features.

📂 Dataset
The Iris dataset is publicly available and can be loaded directly from sklearn.datasets.

🛠 Tech Stack
Programming Language: Python
Libraries Used:
numpy – For numerical computations
pandas – For data manipulation
matplotlib & seaborn – For data visualization
scikit-learn – For machine learning algorithms
🚀 Installation & Setup
🔹 Prerequisites
Ensure you have Python installed (>= 3.7). You can install the required dependencies using:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn
🔹 Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/Iris-Classification.git
cd Iris-Classification
🏗 Model Training
1️⃣ Load Dataset
The dataset is loaded from sklearn.datasets.

python
Copy
Edit
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
2️⃣ Data Visualization
Pair plots and histograms are used to understand feature distributions.
3️⃣ Train-Test Split
Splitting the dataset into 80% training and 20% testing:

python
Copy
Edit
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
4️⃣ Model Training
Using a Support Vector Machine (SVM) classifier:

python
Copy
Edit
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)
5️⃣ Model Evaluation
python
Copy
Edit
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
