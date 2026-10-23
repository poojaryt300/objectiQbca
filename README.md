# ObjectIQ — Smart Object Detector 🔍

> **See it. Scan it. Identify it.**

A modern AI-powered object detection web application built with Python, Streamlit, and YOLOv8.
Detects 80 common objects (people, laptops, phones, bottles, chairs, books, cups, etc.) using the pretrained YOLOv8n model.

---

## 🚀 Features

- 📷 **Camera Scan** – Capture images from your webcam for instant detection
- 🖼️ **Image Upload** – Upload JPG, PNG, JPEG, or WEBP images for analysis
- 🎯 **YOLOv8 AI** – Pretrained on COCO (80 categories), runs fully offline
- 📊 **Results Dashboard** – Metric cards, bar charts, and downloadable CSV reports
- 🎚️ **Confidence Slider** – Adjust detection sensitivity on the fly
- 🌙 **Dark Theme** – Professional navy/purple gradient design

---

## ⚠️ Why Netlify Does NOT Work

**Netlify is designed for static websites** (HTML/CSS/JS files only). It cannot run Python, install packages, or execute Streamlit. Deploying a Python Streamlit app to Netlify will always show "Page not found."

**The correct free platform for Streamlit apps is → [Streamlit Community Cloud](https://streamlit.io/cloud)**

---

## 🛠️ Local Setup (Windows)

### Step 1 – Install Python

Make sure Python 3.9 or higher is installed.
Download from: https://www.python.org/downloads/
✅ During install, check **"Add Python to PATH"**

### Step 2 – Open a Terminal in Your Project Folder

In Windows Explorer, navigate to the project folder, then:
- Right-click inside the folder → **"Open in Terminal"** or **"Open PowerShell window here"**

Or use Command Prompt:
```cmd
cd "c:\Users\Tharun\Downloads\app bca"
```

### Step 3 – Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4 – Activate the Virtual Environment

**PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

If you get a script execution error in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

### Step 5 – Install Dependencies

```bash
pip install -r requirements.txt
```

> This takes 2–5 minutes. The YOLOv8n model file (`yolov8n.pt`) is already included in the project — no download needed.

### Step 6 – Launch the Application

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

Streamlit Community Cloud is the official, **completely free** hosting platform for Streamlit Python apps.

### Step 1 – Push Your Project to GitHub

1. Go to [github.com](https://github.com) and sign in (create a free account if needed)
2. Click **"New repository"** (top-right `+` icon → New repository)
3. Name it: `objectiq-detector`
4. Set it to **Public**
5. Click **"Create repository"**

Now upload your files. You can use GitHub Desktop (easiest) or the command line:

**Option A — GitHub Desktop (Beginner-friendly):**
1. Download [GitHub Desktop](https://desktop.github.com/)
2. Sign in → File → Add Local Repository → browse to your `app bca` folder
3. Commit all files → Push to GitHub

**Option B — Command line:**
```bash
git init
git add app.py requirements.txt README.md .gitignore yolov8n.pt
git commit -m "Initial commit: ObjectIQ Smart Object Detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/objectiq-detector.git
git push -u origin main
```

> ⚠️ **Important:** Make sure `yolov8n.pt` is included in the push. The `.gitignore` has been updated to allow it.

### Step 2 – Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Fill in the form:
   - **Repository:** `YOUR_USERNAME/objectiq-detector`
   - **Branch:** `main`
   - **Main file path:** `app.py`  ← this is the entry point
5. Click **"Deploy!"**

Streamlit Cloud will:
- Clone your GitHub repository
- Install all packages from `requirements.txt` automatically
- Start the app and give you a public URL like:
  `https://YOUR_USERNAME-objectiq-detector-app-xxxxx.streamlit.app`

### Step 3 – Test Your Deployed App

1. Open the public URL from Streamlit Cloud
2. Click **"Upload Image"** tab → upload any photo
3. Click **"Detect Objects"** → view results, confidence scores, bar chart
4. Test **"Camera Scan"** (allow camera permission in browser)
5. Download the CSV report

---

## 📷 Camera Permissions

When using **Camera Scan** mode:
1. Your browser will ask for camera permission — click **Allow**
2. If blocked, click the camera 🎥 icon in the browser address bar → Allow → Refresh
3. Streamlit Cloud serves over HTTPS so camera access works in all modern browsers

**Chrome:** `chrome://settings/content/camera` → allow the app domain

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Netlify shows "Page not found" | Use Streamlit Community Cloud instead — Netlify does not support Python |
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python-headless` |
| Model file not found on Cloud | Ensure `yolov8n.pt` was committed to GitHub (check repo files online) |
| Camera not working locally | Use Chrome, allow camera in browser settings |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Streamlit Cloud build fails | Check the Logs tab in Streamlit Cloud dashboard for the exact error |
| `typing_extensions` conflict | `pip install --upgrade typing_extensions` |

---

## 📁 Project Structure

```
app bca/
├── app.py              ← Main Streamlit application (entry point)
├── requirements.txt    ← Python dependencies
├── yolov8n.pt          ← Pretrained YOLO model (must be in repo for Cloud)
├── README.md           ← This file
└── .gitignore          ← Git ignore rules
```

---

## 🧠 How It Works

1. **Image Input** – Upload an image or capture one from your webcam
2. **Preprocessing** – Pillow converts the image to RGB; OpenCV handles annotation
3. **YOLOv8 Inference** – The neural network detects objects and their bounding boxes
4. **Results** – Displayed with coloured bounding boxes, labels, and confidence %
5. **Report** – Summary dashboard with counts, bar chart, and downloadable CSV

---

## 🎓 About

**ObjectIQ** is a college mini-project demonstrating real-world AI object detection.
Built with ❤️ using Python, Streamlit, and Ultralytics YOLOv8.

| Detail | Value |
|--------|-------|
| Model | YOLOv8n (nano) — fast and lightweight |
| Dataset | Pretrained on COCO (80 object categories) |
| API key | Not required |
| Internet | Only needed for first setup; runs offline after |
