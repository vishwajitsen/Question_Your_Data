#!/usr/bin/env bash
set -e


ROOT="C:/Users/win11/OneDrive/Documents/MiSpy Documents/Question_Your_Data"
cd "${ROOT}"


# Create folders
mkdir -p data src research


echo "Creating virtualenv (venv)..."
python -m venv venv


# activate (unix shell); on Windows Powershell run: .\\venv\\Scripts\\Activate.ps1
source venv/bin/activate || source venv/Scripts/activate


pip install --upgrade pip
pip install -r requirements.txt


# create minimal files if missing
touch src/__init__.py src/helper.py src/prompt.py research/trials.ipynb


cat > .env.example <<EOL
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_REGION=us-east-1
PINECONE_INDEX_NAME=medical-chatbot
EOL


echo "Setup finished. Put PDFs into the 'data/' folder and copy .env.example to .env with your keys."


echo "Run the app:"
echo " streamlit run app.py"