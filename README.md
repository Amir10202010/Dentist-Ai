<div align="center">

# Dentist-Ai

**A Flask web app that runs a custom-trained YOLO model over dental X-rays and returns one annotated image per detected finding.**

Upload a radiograph, get back the original image with boxes drawn around each finding — split into a separate file per class, so crowns, fillings and lesions can be reviewed one layer at a time. Around the detector sits a small clinic-portal shell: registration, login, and a set of marketing and dashboard pages.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-web-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-0B23F5?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-database-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-annotation-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

</div>

## What it does

- **Detection.** `POST /predict` takes an uploaded image, runs the bundled YOLO checkpoint (`app/model/best.pt`) at confidence `0.25` on CPU, and returns JSON with the paths of the generated result images.
- **One image per class.** Instead of a single cluttered overlay, the app writes a separate annotated copy for every class that had at least one detection, named after the class.
- **10 trained classes.** Read from the checkpoint itself; labels are in Russian, as trained: Кариес (caries), Коронка (crown), Пломба (filling), Имплант (implant), Отсутствующие зубы (missing teeth), Периапикальное поражение (periapical lesion), Лечение корневого канала (root canal treatment), Осколок корня (root fragment), Ретинированный зуб (impacted tooth), Потеря костной ткани (bone loss). No accuracy figures are claimed here — none were measured in this repo.
- **Accounts.** Register and log in against PostgreSQL, with email-format validation, an 8-character minimum, `werkzeug` password hashing and a session-cookie `login_required` guard.
- **Pages.** A Russian-language landing page plus About, Contacts, Pricing, a privacy-policy page, and a dashboard shell behind login.

## How it works

A request to `/predict` (registered as a Flask Blueprint in `app/predict.py`) flows like this:

1. The multipart field `image` is saved through `secure_filename()` into `app/static/uploads/`.
2. `model.predict(..., conf=0.25, device="cpu")` runs the checkpoint over that file.
3. A new output folder is created at `app/static/predicts/resN/`, where `N` is the number of existing `res*` folders plus one.
4. For each class with at least one box, the original image is copied, every box for that class is drawn on the copy with `cv2.rectangle`, and the copy is written to `resN/<class name>.jpg`.
5. The response is `{"success": true, "results": [...]}`, or `{"success": false, "message": "Патологии не обнаружены"}` when nothing crossed the threshold.

```
app/
├── main.py              # Flask app: pages, session auth, /api/* endpoints
├── predict.py           # Blueprint: loads best.pt, detects, writes annotated images
├── model/best.pt        # custom-trained YOLO checkpoint (~52 MB, committed)
├── templates/           # Jinja2 pages + header/footer/sidebar partials
└── static/
    ├── css/, js/        # one stylesheet and one script per page
    ├── uploads/         # images posted to /predict land here
    └── predicts/resN/   # one annotated image per detected class
```

## Running locally

Requires Python 3.10+ and a reachable PostgreSQL instance. The model file `app/model/best.pt` is already committed, so there is nothing to download.

```bash
git clone https://github.com/Amir10202010/Dentist-Ai.git
cd Dentist-Ai
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `app/.env`:

```
DB_HOST=localhost
DB_NAME=dentist_ai
DB_USER=postgres
DB_PASSWORD=your_password
FLASK_SECRET_KEY=some_long_random_string
```

The app resolves both the model path and its `.env` relative to the working directory, so start it from inside `app/`:

```bash
cd app
python main.py     # http://127.0.0.1:5001
```

No schema file is committed. The queries in `main.py` expect two tables — `users(id, name, email, password)` and `ai_history(id, user_id, query, response, created_at)` — which you have to create yourself before registration or history will work.

The detector has no UI wired to it, so exercise it directly:

```bash
curl -F "image=@app/static/uploads/00005.jpg" http://127.0.0.1:5001/predict
```

## Status

A student prototype, not a product, and definitely not a medical device — nothing here has been clinically validated.

What is actually finished: the `/predict` detection endpoint, and register/login/logout against Postgres. What is not:

- `/predict` is reachable by API only. The dashboard's file input just prints "анализ… (демо)" after a timer and never calls it.
- The dashboard's KPIs, charts, patient table and calendar are hard-coded demo data in `static/js/home.js`.
- `/api/ai/run` stores a fixed placeholder string instead of calling a model.
- `/calendar`, `/analytics` and `/settings` render templates that do not exist, and `patients.html` is an empty file — those routes error.
- The contact form posts to `/send_message`, which is not implemented.
- Inference runs on CPU (`device="cpu"` is hard-coded), so a full-size panoramic X-ray takes a few seconds per request.
- `app.run(debug=True)` and the `"dev-secret"` fallback session key are development settings — replace both before exposing this anywhere.

---

Built by [Amirkhan Sagyndyk](https://github.com/Amir10202010).
