# Using Machine Learning to Determine the Half-Lives of Radioisotopes
# CHE1147 Term Project

## Project Overview
This project uses machine learning techniques to predict the half-lives of radioisotopes using selected nuclear properties. Multiple models were developed and evaluated to investigate the relationships between nuclear features and radioactive decay. A key focus of this project was investigating different approaches to handling missing data and evaluating their impact on model performance.

## ⚙️ Installation & Usage

### 1. Clone the Repository

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/BrandonL08/Nuclear-Half-Life-Prediction-CHE1147-Term-Project.git
cd Nuclear-Half-Life-Prediction-CHE1147-Term-Project
pip install -r requirements.txt
```

### 2. Running the Notebooks

The notebooks in the `Models` folder should be run in the following order:

1. **Included_NaN** – Models trained using datasets with missing values retained.
2. **dropped_NaN** – Models trained after removing records containing missing values.
3. **imputed_NaN** – Models trained using median imputation to handle missing values.
4. **Comparison** – Compares the best-performing models from each missing-data approach.

### 3. Additional Notes

- The cleaned dataset (`cleaned_data.csv`) is already provided, so running the files in the `Extraction/` folder is not necessary.
- Run the notebooks in the specified order to ensure the comparison notebook has the necessary results.

## 📁 Repository Structure

📂 EDA/ – Contains notebooks associated with the EDA and data cleaning. Generated figures are included in a sub-folder.

📂 Extraction/ – Contains files related to extracting and preparing the dataset from the original ZIP folder. It is NOT necessary to run these files as the dataset has already been cloned to cleaned_data.csv

📂 Models/ – Contains machine learning models, training and evaluation. The models are divided into four files: three separate approaches to handling missing data + a comparison of the best models from each   approach. 

📂 Python Scripts/ – Contains supporting Python scripts used throughout the project.

📄 README.md – Provides an overview of the project and repository.

📄 requirements.txt – Lists the packages required to run the project.

📄 pyproject.toml – Contains project configuration and dependency information.

📄 uv.lock – Records the exact dependency versions used in the project.

## Group Members
Brayden Hutchinson, JinWon Choi, Brandon Lau
