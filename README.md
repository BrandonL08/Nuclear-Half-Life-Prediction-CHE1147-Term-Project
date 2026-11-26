# Using Machine Learning to Determine the Half-Lives of Radioisotopes
# CHE1147 Term Project

## Project Overview
This project uses machine learning techniques to predict the half-lives of radioisotopes using selected nuclear properties. Multiple models were developed and evaluated to investigate the relationships between nuclear features and radioactive decay. A key focus of this project was investigating different approaches to handling missing data and their impact on model performance. These approaches include dropping, including, or imputing missing values. Data for 3,371 isotopes were extracted from the Evaluated Nuclear Structure Data File (ENSDF), courtesy of the Brookhaven National Laboratory and the United States Department of Energy.

## ⚙️ Installation & Instructions

### 1. Clone the Repository

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/BrandonL08/Nuclear-Half-Life-Prediction-CHE1147-Term-Project.git
cd Nuclear-Half-Life-Prediction-CHE1147-Term-Project
pip install -r requirements.txt
```

### 2. Running the Models

The notebooks in the `Models` folder should be run in the following order:

1. **Included_NaN** – Models trained using datasets with missing values retained.
2. **dropped_NaN** – Models trained after removing records containing missing values.
3. **imputed_NaN** – Models trained using median imputation to handle missing values.
4. **Comparison** – Compares the best-performing models from each missing-data approach.

### 3. Additional Notes
- Python Scripts were included in a separate folder if you would prefer to run the models that way
- EDA, and generated figures are included in the `EDA` folder
- The cleaned dataset (`cleaned_data.csv`) is already provided, so **running the files in the `Extraction` folder is not necessary**.
- Run the notebooks in the specified order to ensure the comparison notebook has the necessary results.

## 3. 📁 Repository Structure

📂 EDA/ – Contains notebooks associated with the EDA and data cleaning. Generated figures are included in a sub-folder.

📂 Extraction/ – Contains files related to extracting and preparing the dataset from the original ZIP folder. It is NOT necessary to run these files as the dataset has already been cloned to cleaned_data.csv

📂 Models/ – Contains machine learning models, training and evaluation. The models are divided into four files: three separate approaches to handling missing data + a comparison of the best models from each   approach. 

📂 Python Scripts/ – Contains a copy of the code in script form rather than notebook form.

📄 README.md – Provides an overview of the project and repository.

📄 requirements.txt – Lists the packages required to run the project.

📄 pyproject.toml – Contains configuration information.

📄 uv.lock – Records the exact versions of the packages used in the project.

## Group Members
Brayden Hutchinson, JinWon Choi, Brandon Lau
