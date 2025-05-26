# Machine Learning based Energy Consumption Prediction
<p align="center">
  <img src="https://github.com/akhilchibber/Energy-Consumption-Prediction/blob/main/Energy-Consumption.png?raw=true" alt="earthml Logo">
</p>

## Dataset
The dataset used in this project can be found on Kaggle: [Energy Consumption Dataset](https://www.kaggle.com/competitions/ashrae-energy-prediction/data). 

## Getting Started
To get started with this project:

1. Clone this repository to your local machine.
2. Ensure you have Jupyter Notebook installed and running (for exploring the original notebook).
3. Install the required dependencies (see "Frontend Application for Prediction" section for details).
4. Download the "Energy Consumption Dataset" and place it in the designated directory if you intend to retrain models.
5. Open and run the Jupyter Notebook "Energy-Consumption-Prediction.ipynb" to explore the original model training and evaluation. For the refactored pipeline and prediction, see the sections below.

## Frontend Application and Prediction Pipeline

This project includes a Streamlit frontend application (`app.py`) and a command-line script (`predict_energy.py`) for generating energy consumption predictions using pre-trained models.

### a. Setup
Install the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

### b. Model Training (Important Prerequisite for Real Predictions)
To generate actual prediction models, the `Energy-Consumption-Prediction-Pipeline.ipynb` notebook must be run with the original competition data.

1.  **Required Data:**
    You need the original ASHRAE Great Energy Predictor III Kaggle competition data:
    *   `train.csv`
    *   `building_metadata.csv`
    *   `weather_train.csv`
    *   `rows_to_drop.csv` (if used by the pipeline notebook for training - this file was part of the original Kaggle dataset or provided by the user)

2.  **Data Placement:**
    Place these files in a directory structure that the notebook expects. The notebook is configured to look for data in `./kaggle/input/` if a `/kaggle/input` directory (common in Kaggle environments) is not found. For example:
    ```
    ./kaggle/input/ashrae-energy-prediction/train.csv
    ./kaggle/input/ashrae-energy-prediction/building_metadata.csv
    ./kaggle/input/ashrae-energy-prediction/weather_train.csv
    ./kaggle/input/rows-to-drop/rows_to_drop.csv 
    ```
    Alternatively, update the paths directly in the `Energy-Consumption-Prediction-Pipeline.ipynb` notebook.

3.  **Run Training Notebook:**
    *   Open `Energy-Consumption-Prediction-Pipeline.ipynb` in a Jupyter Notebook or JupyterLab environment.
    *   Ensure the `TRAIN_MODELS` flag (a Python variable in one of the initial code cells) is set to `True`.
    *   Run all cells in the notebook. This process will train 5 LightGBM models (one for each fold) and save them as `lgbm_model_fold_0.txt`, `lgbm_model_fold_1.txt`, ..., `lgbm_model_fold_4.txt` in the same directory as the notebook.

    **Note:** Without running this training step with the full data, the application and prediction script will attempt to use dummy/placeholder models if the `DUMMY_MODEL_MODE` environment variable is set to `"true"` in `predict_energy.py` (allowing for pipeline testing), or they will fail to load models if real model files are not found.

### c. Running the Streamlit Prediction Application
Once models are trained and saved (or if you intend to test the pipeline with dummy models):

1.  **Model Placement:** Ensure the generated model files (`lgbm_model_fold_*.txt`) are in the same directory as `app.py` and `predict_energy.py`. The prediction script looks for models in its current working directory by default.
2.  **Run Streamlit App:** Open your terminal, navigate to the project directory, and run:
    ```bash
    streamlit run app.py
    ```
3.  **Access Application:** Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
4.  **Upload Data:** Upload the three required CSV files for prediction:
    *   Your `test.csv` file.
    *   The corresponding `building_metadata.csv`.
    *   The corresponding `weather_test.csv`.
5.  **Generate Predictions:** Click the "Generate Predictions" button.
6.  **Download Results:** A download button for the generated `submission.csv` file will appear once processing is complete.

### d. (Optional) Using `predict_energy.py` directly (CLI)
Alternatively, you can use the command-line script `predict_energy.py` directly for predictions. This also requires the trained model files (`lgbm_model_fold_*.txt`) to be accessible (e.g., in the same directory or specified via the `--model_dir` argument).

**Usage:**
```bash
python predict_energy.py path/to/your/test.csv path/to/your/building_metadata.csv path/to/your/weather_test.csv path/to/your/output_submission.csv --model_dir path/to/your/models
```
Replace the placeholder paths with the actual paths to your files and desired output location. If `--model_dir` is omitted, it defaults to the current directory.

## Contributing
We welcome contributions to enhance the functionality and efficiency of this script. Feel free to fork, modify, and make pull requests to this repository. To contribute:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request against the `main` branch.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

Author: Akhil Chhibber

LinkedIn: https://www.linkedin.com/in/akhilchhibber/

Medium Blogs: https://medium.com/@akhil.chibber
