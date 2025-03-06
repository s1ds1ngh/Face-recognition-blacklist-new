# Real Time Face Recognition System

## How to run this project

1. Activate your virtual environment
```bash
source venv/bin/activate
```
2. Install the required packages using the following command
```bash
pip install -r requirements.txt
```
3. Run the following command to start the flask application

`Make sure to replace the IP address in the RSTP URL in constants.py`
```bash
python app.py
```
4. Run the following command to start the streamlit application
```bash
streamlit run streamlit_app.py
```

**Note**:
Run either flask application or streamlit application at a time. Running both applications at the same time will cause a conflict.