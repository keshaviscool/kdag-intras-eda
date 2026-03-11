# EDA Dashboard

Exploratory Data Analysis dashboard for the KDAG Intras '26 dataset.

## Local Run
```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5050

## Deploy on Render (free)
1. Push this `eda_dashboard/` folder to a GitHub repo.
2. Go to https://render.com → New → Web Service.
3. Connect your GitHub repo.
4. Set **Root Directory** to `eda_dashboard` (if it's inside a larger repo).
5. Set **Build Command**: `pip install -r requirements.txt`
6. Set **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
7. Choose the **Free** plan and deploy.

Your dashboard will be live at `https://your-app.onrender.com`.
