import streamlit as st
import pandas as pd
import os
import tempfile # For robust temporary directory creation
import shutil # For cleaning up temporary directory

# Attempt to import the prediction function
# This assumes predict_energy.py is in the same directory or PYTHONPATH is set up
try:
    from predict_energy import generate_predictions
except ImportError:
    st.error("Failed to import 'generate_predictions' from 'predict_energy.py'. "
             "Ensure the script is in the same directory and all its dependencies are met.")
    # Stop further execution if import fails
    st.stop()

# --- Streamlit App UI ---
st.title("Energy Consumption Prediction")

st.markdown("""
Upload the required CSV files to generate energy consumption predictions.
Model files (`lgbm_model_fold_*.txt`) are assumed to be in the same directory as `predict_energy.py` or a path `predict_energy.py` can access.
""")

# File Uploaders
uploaded_test_file = st.file_uploader("Upload Test Data (test.csv)", type="csv")
uploaded_building_file = st.file_uploader("Upload Building Metadata (building_metadata.csv)", type="csv")
uploaded_weather_file = st.file_uploader("Upload Weather Data (weather_test.csv)", type="csv")

# Predict Button
if st.button("Generate Predictions"):
    # Input Validation
    if uploaded_test_file is not None and uploaded_building_file is not None and uploaded_weather_file is not None:
        
        # Create a temporary directory
        # tempfile.mkdtemp() creates a unique directory
        temp_dir = tempfile.mkdtemp(prefix="streamlit_pred_")
        
        try:
            st.info("Processing... please wait. This may take several minutes depending on the data size.")

            # Define paths for temporary files
            test_csv_path = os.path.join(temp_dir, "test.csv")
            building_csv_path = os.path.join(temp_dir, "building_metadata.csv")
            weather_csv_path = os.path.join(temp_dir, "weather_test.csv")
            output_submission_path = os.path.join(temp_dir, "submission.csv")
            
            # Save uploaded files to the temporary directory
            with open(test_csv_path, "wb") as f:
                f.write(uploaded_test_file.getbuffer())
            with open(building_csv_path, "wb") as f:
                f.write(uploaded_building_file.getbuffer())
            with open(weather_csv_path, "wb") as f:
                f.write(uploaded_weather_file.getbuffer())

            # Call Prediction Script
            # Assuming model files are in the same directory as predict_energy.py (which is usually where app.py is)
            # or predict_energy.py handles the model_dir internally if models are elsewhere.
            # For this setup, predict_energy.py's model_dir default is "." (current dir)
            generate_predictions(
                test_csv_path=test_csv_path,
                building_metadata_csv_path=building_csv_path,
                weather_test_csv_path=weather_csv_path,
                output_csv_path=output_submission_path,
                model_dir="." # Explicitly state model dir, or remove if predict_energy.py handles it well
            )
            
            # Provide Download Link
            if os.path.exists(output_submission_path):
                st.success("Predictions generated successfully!")
                with open(output_submission_path, "rb") as f:
                    st.download_button(
                        label="Download Predictions (submission.csv)",
                        data=f,
                        file_name="submission.csv",
                        mime="text/csv",
                    )
            else:
                st.error("Prediction script ran, but the submission file was not generated.")

        except ImportError as ie:
             st.error(f"Import Error: {ie}. Please ensure all dependencies for predict_energy.py are installed (check requirements.txt).")
        except FileNotFoundError as fnfe:
            st.error(f"File Not Found Error during prediction: {fnfe}. This could be due to missing model files expected by predict_energy.py.")
        except Exception as e:
            st.error(f"An error occurred during the prediction process: {e}")
            # For more detailed debugging, you might want to log the full traceback
            # import traceback
            # st.text(traceback.format_exc())
        finally:
            # Cleanup: Remove the temporary directory and its contents
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
                
    else:
        st.error("Please upload all three required CSV files.")

st.markdown("---")
st.markdown("Developed as part of a software engineering task.")
