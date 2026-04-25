from flask import Flask, render_template, request, jsonify, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import psycopg2
from dotenv import load_dotenv
import os
import re
from predict import predict_bp

load_dotenv()

app = Flask(__name__)
app.register_blueprint(predict_bp)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# ================== DATABASE ==================
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

# ================== HELPERS ==================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper

def is_valid_email(email):
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email)

# ================== PAGES ==================
@app.route("/")
def main():
    return render_template("main.html")

@app.route("/login")
def login():
    if "user_id" in session:
        return redirect("/home")
    return render_template("login.html")

@app.route("/register")
def register():
    if "user_id" in session:
        return redirect("/home")
    return render_template("register.html")

@app.route("/home")
@login_required
def home():
    return render_template("home.html")

# ================== AUTH ==================
@app.route("/register", methods=["POST"])
def register_user():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")

    if not all([name, email, password, confirm]):
        return jsonify({"status": "danger", "message": "Все поля обязательны"})

    if not is_valid_email(email):
        return jsonify({"status": "danger", "message": "Неверный email"})

    if len(password) < 8:
        return jsonify({"status": "danger", "message": "Пароль минимум 8 символов"})

    if password != confirm:
        return jsonify({"status": "danger", "message": "Пароли не совпадают"})

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            return jsonify({"status": "danger", "message": "Email уже используется"})

        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (%s,%s,%s)",
            (name, email, generate_password_hash(password))
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"status": "success", "message": "Регистрация успешна"})

    except Exception as e:
        return jsonify({"status": "danger", "message": str(e)})

@app.route("/login", methods=["POST"])
def login_user():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password, name FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if not user or not check_password_hash(user[1], password):
            return jsonify({"status": "danger", "message": "Неверные данные"})

        session["user_id"] = user[0]
        session["user_name"] = user[2]

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "danger", "message": str(e)})

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================== API ==================
@app.route("/api/user/info")
@login_required
def api_user_info():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, email FROM users WHERE id=%s",
            (session["user_id"],)
        )
        user = cur.fetchone()
        cur.close()
        conn.close()

        if not user:
            return jsonify({"status": "danger", "message": "User not found"})

        return jsonify({
            "status": "success",
            "user": {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
        })

    except Exception as e:
        return jsonify({"status": "danger", "message": str(e)})

@app.route("/api/user/history")
@login_required
def api_user_history():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT query, response, created_at
            FROM ai_history
            WHERE user_id=%s
            ORDER BY created_at DESC
            LIMIT 20
        """, (session["user_id"],))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        history = [{
            "query": r[0],
            "response": r[1],
            "created_at": r[2]
        } for r in rows]

        return jsonify({"status": "success", "history": history})

    except Exception as e:
        return jsonify({"status": "danger", "message": str(e)})

@app.route("/api/ai/run", methods=["POST"])
@login_required
def api_ai_run():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"status": "danger", "message": "Пустой запрос"})

    answer = "Это ответ от AI (заглушка)"

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ai_history (user_id, query, response)
            VALUES (%s,%s,%s)
        """, (session["user_id"], prompt, answer))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("AI history error:", e)

    return jsonify({"status": "success", "answer": answer})

@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    if "file" not in request.files:
        return jsonify({"status": "danger", "message": "Нет файла"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "danger", "message": "Пустое имя файла"})

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", f"{session['user_id']}_{file.filename}")
    file.save(path)

    return jsonify({"status": "success"})
    
@app.route("/patients")
@login_required
def patients():
    return render_template("patients.html")  # можно пока пустой файл

@app.route("/calendar")
@login_required
def calendar():
    return render_template("calendar.html")

@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html")

@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/doc')
def doc():
    return render_template('doc.html')

# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True, port=5001)
