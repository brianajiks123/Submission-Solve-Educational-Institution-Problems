# Final Project: Solving Edutech Company Problems

## Business Understanding

Jaya Jaya Institute has been operating since 2000 and has a strong reputation for producing quality graduates. However, the dropout rate remains high and remains a major concern. To maintain its reputation and improve the quality of its educational services, the institution aims to better understand the factors influencing student performance and identify students at risk of dropping out early. With a data-driven approach, the institution hopes to provide timely intervention through specialized guidance to reduce the dropout rate.

### Business Problems

Based on the background above, the main problems to be addressed are:

1. The high number of students dropping out at Jaya Jaya Institute negatively impacts the institution's reputation.

2. The absence of a predictive system that can identify students at risk of dropping out early.

3. Difficulty in comprehensively understanding and monitoring student performance due to scattered and poorly integrated data.

4. Lack of appropriate and timely intervention due to the lack of real-time information regarding student conditions and performance.

5. Limitations in data-driven decision-making, which can hinder effective dropout prevention efforts.

### Project Scope

To address the above issues, this project includes five integrated phases as follows.

1. Data Collection & Cleaning

    - Reading and verifying the format and completeness of the dataset (data.csv).
    - Handling missing values and duplication.
    - Detecting and handling outliers using the IQR method.
    - Transforming the target variable Status into binary format (Status_binary) for classification purposes.

    Main output: Clean dataset ready for analysis and modeling.

2. Exploratory Data Analysis (EDA)

    - Analysis of the distribution of categorical and numeric variables.
    - Visualization of the relationship between features and dropout status.
    - Correlation heatmap between numeric features.
    - Identification and removal of highly correlated features (>0.95).

    Main output: Data exploration report containing initial insights into student characteristics and dropout indicators.

3. Predictive Dropout Modeling

    - Development of a preprocessing pipeline (imputation, scaling, encoding, outlier clipping, feature selection).

    - Model training and evaluation using the following algorithms:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Gradient Boosting
    - Random Forest
    - XGBoost
    - LightGBM
    - Multi-layer Perceptron (MLP)
    - Model evaluation based on AUC and F1-score metrics.
    - Classification threshold adjustment based on precision-recall.
    - Visualization of the most influential features on predictions.

    Main outputs: Best dropout predictive model, saved model file (best_dropout_model.joblib), and complete performance evaluation.

4. Business Dashboard Design & Development

    - Wireframe & Prototype Creation
    - Visualization Integration in Metabase

    Main Output: Web-based interactive dashboard

5. Documentation

    - README.md Creation

    Main Output: README file containing project documentation.

### Preparation

Data source: [Employee Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

Setup environment (python):

1. Creating a Virtual Environment (Windows)

    ``` python
    python -m venv venv
    ```

    ``` bash
    .\venv\Scripts\activate
    ```

2. Install Library

    ``` python
    pip install -r requirements.txt
    ```

3. Running Metabase (Business Dashboard)

    - Make sure you have Docker Desktop installed
    - Run Powershell and go to the project directory
    - Create a PostgreSQL Network

    ``` bash
    docker network create pg-network 
    ```

    - Create a Database Container (Postgresql)

    ``` bash
    docker run --name academic-postgres --network=pg-network -e POSTGRES_PASSWORD=academic -e POSTGRES_DB=academic -p 5432:5432 -d postgres 
    ```

    - Loading Employee Dataset into Database

    ``` bash
    python academic.py 
    ```

    - Run Container Metabase + Load Metabase Database

    ``` bash
    docker run -d -p 3000:3000 --name academic-metabase -v "$(pwd)/metabase.db:/tmp" -e "MB_DB_FILE=/tmp/metabase.db" --network=pg-network metabase/metabase 
    ```

4. Run Jupyter Notebook (Modeling) -> Optional

    - Run Powershell and navigate to the project directory

    ``` bash
    jupyter lab
    ```

## Business Dashboard

!["Section 1"](./business%20dashboard/business%20dashboard_1.png)
!["Section 2"](./business%20dashboard/business%20dashboard_2.png)

Dashboard Key Features:

- Interactive Filter:
  - Course
  - Application Mode
  - Admission Grade
- KPI Summary:
  - Total Active Students
  - Total Dropouts
  - Average Entry Grade / Selection Grade
  - Dropout Percentage
  - Average Applicant Age
- Main Visualization:
  - 5 Study Programs with the Highest Dropouts
  - Admission Path vs. Number of Dropouts
  - Risk Profile: Entry Grade / Selection Grade vs. Courses Without Assessment in Semester 1
  - Financial Status vs. Dropout Rate

### Access Business Dashboard

<http://localhost:3000>

Credentials: user -> <root@mail.com> | password -> root123

## Running Machine Learning Systems

- Run Powershell and go to the project directory

    ``` bash
    streamlit run app.py
    ```

### Prototype Access

[Streamlit](https://proyek-permasalahan-perusahaan-edutech-brianajipamungkas.streamlit.app/)

## Conclusion

After developing the dashboard and conducting predictive modeling, the following findings were obtained:

- All metrics related to semester 2 (approved units, grades, enrolled units, evaluations) occupied the majority of the top positions. This means that student performance and participation in semester 2 are the strongest indicators for predicting dropout.
- "Tuition_fees_up_to_date" and "Debtor" indicate that tuition fee payment issues increase the risk of dropout.
- Units approved in semester 1 and age at enrollment also contributed, albeit to a lesser extent. This suggests the importance of support from the start (semester 1) and potential differences in needs based on age.
- One study program (Course 171) exhibited unique dropout characteristics, requiring a review of its curriculum and support.

### Recommended Actions

Provide several recommended action items that companies should implement to resolve issues or achieve their targets.

- Provide additional academic mentoring, especially in the second semester, for students with a low number of approved units or declining grades.
- Identify and assist students with outstanding fees, for example through scholarships, installments, or financial counseling.
- Orientation and tutoring programs in the first semester to increase the number of approved units early on.
- Investigate the causes of the high dropout rate in Course 171, including the curriculum, assignment load, or supporting facilities.
- Tailor services (e.g., time management workshops) based on age group, as age at enrollment also influences outcomes.
