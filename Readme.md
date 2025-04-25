# Reentrancy Vulnerability Detector (VSCode Plugin)

A VSCode plugin for detecting **reentrancy vulnerabilities** in Ethereum smart contracts. This tool is based on the **VoteRD** deep learning model and provides a right-click detection interface inside VSCode.

> 📌 Version: `v1.0` – Detection only, no training or retraining features.

---

## 🧠 Model Overview

The core detection functionality is powered by the **VoteRD** model, a voting-based deep learning system designed for identifying reentrancy vulnerabilities in Solidity smart contracts. Given a selected code fragment, the model predicts whether it is reentrant or benign, using semantic and positional features.

---

## ⚙️ Environment Setup

This project requires:

- **Python**: `3.7.0`
- **VSCode** with the ability to run `.vsix` extensions
- You need to manually create a virtual environment (folder name: venv) at the project root
- Python Dependencies：
  ```
  absl-py==2.2.2
antlr4-python3-runtime==4.9.3
astunparse==1.6.3
certifi==2025.1.31
charset-normalizer==3.4.1
contourpy==1.3.0
cycler==0.12.1
flatbuffers==25.2.10
fonttools==4.57.0
gast==0.6.0
google-pasta==0.2.0
grpcio==1.71.0
h5py==3.13.0
idna==3.10
importlib_metadata==8.6.1
importlib_resources==6.5.2
joblib==1.4.2
keras==3.9.2
kiwisolver==1.4.7
libclang==18.1.1
Markdown==3.8
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.9.4
mdurl==0.1.2
ml_dtypes==0.5.1
namex==0.0.9
numpy==2.0.2
opt_einsum==3.4.0
optree==0.15.0
packaging==25.0
pandas==2.2.3
pillow==11.2.1
protobuf==5.29.4
Pygments==2.19.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.3
rich==14.0.0
scikit-learn==1.6.1
scipy==1.13.1
six==1.17.0
solidity-parser==0.1.1
tensorboard==2.19.0
tensorboard-data-server==0.7.2
tensorflow==2.19.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==3.0.1
threadpoolctl==3.6.0
typing_extensions==4.13.2
tzdata==2025.2
urllib3==2.4.0
Werkzeug==3.1.3
wrapt==1.17.2
zipp==3.21.0
```
### 🔧 Installation Steps

1. Create and activate a virtual environment:

   **Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Using the Plugin

1. Open your Solidity project in **Visual Studio Code**.
2. Select a code block to analyze.
3. Right-click and choose **“检测 Reentrancy”** from the context menu.
4. The model will run and visually highlight vulnerable lines, if found.

---

## 📦 VSCode Plugin Installation

Download the plugin from the [Releases](https://github.com/你的用户名/你的仓库名/releases) page and install it manually using:

```bash
code --install-extension vscode-scripts-1.0.0.vsix
```

---

## 🧾 License

This project is intended for research and educational purposes only. Use at your own risk.
