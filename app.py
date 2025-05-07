import os
import cv2
import threading
from flask import Flask, url_for, render_template, request, redirect, session, g, flash, get_flashed_messages
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import numpy as np
import sqlite3
from functools import wraps
import datetime
import pytz  # for timezone conversion
from zoneinfo import ZoneInfo  # For Python 3.9+

from live_stream_recognition import recognize_faces
from train_image import train_image

app = Flask(__name__)
app.config["SECRET_KEY"] = "secretkey"

# Directory to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

DATABASE = "db.sqlite3"

haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "./TrainingImageLabel/Trainner.yml"
trainimage_path = "/TrainingImage"
if not os.path.exists(trainimage_path):
    os.makedirs(trainimage_path)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

camera = None  # Global variable to store the camera stream
stream_thread = None  # To track the OpenCV stream thread
streaming = False  # To check if the stream is active


def open_camera_stream(camera_url):
    with app.app_context():
        """Function to open the camera stream in a new window using OpenCV."""
        global streaming
        streaming = True
        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            print("Error: Unable to open camera stream.")
            streaming = False
            return

        # Bring the OpenCV window to the front
        cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera Stream", np.zeros((100, 100, 3), dtype=np.uint8))  # Show a dummy image
        cv2.waitKey(1)  # Give OpenCV some time to create the window
        cv2.setWindowProperty("Camera Stream", cv2.WND_PROP_TOPMOST, 1)

        frame_skip = 8  # Process every 8th frame
        frame_count = 0

        while streaming:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to retrieve frame.")
                break

            frame_count += 1
            if frame_count % frame_skip == 0:
                # Detect faces only on every 'frame_skip' frame
                # frame_with_faces = detect_faces(frame)
                frame_with_faces = recognize_faces(frame, get_db)
                cv2.imshow("Camera Stream", frame_with_faces)

            # Exit the stream when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        streaming = False

# Function to validate allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to get the database connection
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, check_same_thread=False)
        g.cursor = g.db.cursor()
    return g.db, g.cursor

# Function to close the database connection
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize the database
def initialize_database():
    db, c = get_db()

    # Create the 'users' table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        user_name TEXT,
        user_type TEXT NOT NULL
    )
    ''')

    # Generate a hashed password
    hashed_org_password = generate_password_hash("org123", method="pbkdf2:sha256", salt_length=8)

    # Insert the user if they don't already exist
    c.execute('''
    INSERT OR IGNORE INTO users (email, password, user_type)
    VALUES (:email, :password, :user_type)
    ''', {
        "email": "org123@gmail.com",
        "password": hashed_org_password,
        "user_name": "Org Admin",
        "user_type": "organization"
    })

    # Create the 'userImages' table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS userImages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')

    # Create the 'trainingStatus' table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS trainingStatus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        is_trained BOOLEAN NOT NULL DEFAULT 0,
        trained_at TIMESTAMP DEFAULT NULL,
        FOREIGN KEY (image_id) REFERENCES userImages (id) ON DELETE CASCADE
    )
    ''')

    # Create attendance table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        status TEXT CHECK(status IN ('entered', 'exited')) NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')


    db.commit()

# Initialize the database at application start
with app.app_context():
    initialize_database()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # If the user is not logged in, redirect to the login page
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    db, c = get_db();
    signupErrors = ''

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")
        # check if the form is valid

        if not email or not password or not confirmation:
            signupErrors = "please fill out all fields"

        if password != confirmation:
            signupErrors = "password confirmation doesn't match password"

        # check if email exist in the database
        exist = c.execute("SELECT * FROM users WHERE email=:email", {"email": email}).fetchall()

        if len(exist) != 0:
            signupErrors = "user already registered"
        else:
            # hash the password
            pwhash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=8)

            # insert the row
            c.execute("INSERT INTO users (email, password, user_type) VALUES (:email, :password, :user_type)", {"email": email, "password": pwhash, "user_type": "employee"})
            db.commit()

            return render_template('login.html');
    
    return render_template("register.html", signupErrors = signupErrors)


@app.route("/login", methods=["GET", "POST"])
def login():
    db, c = get_db();
    loginErrors = ''  # Initialize variable for error messages

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Check if required fields are filled
        if not email or not password:
            loginErrors = "Please fill out all required fields."
        else:
            # Check if the email exists in the database
            user = c.execute("SELECT * FROM users WHERE email=:email", {"email": email}).fetchone()

            if not user:
                loginErrors = "Email not registered. Please register first."
            else:
                # Validate the password
                stored_password_hash = user[2]  # Assuming the password hash is stored in the third column
                if not check_password_hash(stored_password_hash, password):
                    loginErrors = "Invalid password. Please try again."
                else:
                    # Login the user
                    session["user_id"] = user[0]  # Assuming the user ID is in the first column
                    session["email"] = user[1]
                    session["user_name"] = user[3]
                    session["user_type"] = user[4]

                    return redirect("/dashboard")  # Redirect to employee dashboard

    return render_template("login.html", loginErrors=loginErrors)

@app.route("/logout",  methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/about")
def about():
     return render_template("about.html")

@app.route("/how-to-use")
def howToUse():
    return render_template("how-to-use.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/dashboard")
@login_required
def userDashboard():
    db, c = get_db()
    user_type = session['user_type']
    user_id = session['user_id']
    
    time_limit = datetime.timedelta(minutes=10)
    
    # Get selected date from query string
    selected_date_str = request.args.get('date')
    
    if selected_date_str:
        selected_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").date()
    else:
        selected_date = datetime.date.today()

    start_datetime = datetime.datetime.combine(selected_date, datetime.datetime.min.time())
    end_datetime = datetime.datetime.combine(selected_date, datetime.datetime.max.time())

    present_status = "N/A"  # Ensure it is initialized before any logic

    if user_type == 'employee':
        c.execute("""
            SELECT timestamp, status
            FROM attendance 
            WHERE user_id = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (user_id, start_datetime, end_datetime))

        records = c.fetchall()

        total_duration = datetime.timedelta()
        current_entry = None
        
        LOCAL_TZ = ZoneInfo('Asia/Kolkata')

        first_entry_str = 'N/A'
        last_exit_str = 'N/A'

        for timestamp_str, status in records:
            dt_utc = datetime.datetime.fromisoformat(timestamp_str).replace(tzinfo=ZoneInfo('UTC'))
            dt_local = dt_utc.astimezone(LOCAL_TZ)

            if status == 'entered':
                if first_entry_str == 'N/A':
                    first_entry_str = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                current_entry = dt_local
            elif status == 'exited' and current_entry:
                duration = dt_local - current_entry
                total_duration += duration
                last_exit_str = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                current_entry = None

        if current_entry:
            duration_str = 'Not exited'
        else:
            total_seconds = int(total_duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            duration_str = f"{hours}h {minutes}m {seconds}s"

        if total_duration.total_seconds() > 0:
            if total_duration.total_seconds() >= time_limit.total_seconds():
                present_status = "Present"
            else:
                present_status = "Half Day"

        return render_template("user-templates/user-dashboard.html", 
                               date=selected_date, 
                               selected_date=selected_date_str,
                               first_entry=first_entry_str, 
                               last_exit=last_exit_str, 
                               duration=duration_str,
                               present=present_status)
    else:
        c.execute("""
            SELECT u.user_name, a.timestamp, a.status
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            WHERE a.timestamp BETWEEN ? AND ?
            ORDER BY u.user_name, a.timestamp ASC
        """, (start_datetime, end_datetime))

        records = c.fetchall()
        summary = []
        user_sessions = {}

        for user_name, timestamp, status in records:
            if user_name not in user_sessions:
                user_sessions[user_name] = []
            user_sessions[user_name].append((timestamp, status))

        LOCAL_TZ = ZoneInfo('Asia/Kolkata')

        for user_name, sessions in user_sessions.items():
            total_duration = datetime.timedelta()
            first_entry_str = 'N/A'
            last_exit_str = 'N/A'
            current_entry = None

            for timestamp, status in sessions:
                dt_utc = datetime.datetime.fromisoformat(timestamp).replace(tzinfo=ZoneInfo('UTC'))
                dt_local = dt_utc.astimezone(LOCAL_TZ)

                if status == 'entered':
                    if not current_entry:
                        current_entry = dt_local
                        if first_entry_str == 'N/A':
                            first_entry_str = current_entry.strftime('%Y-%m-%d %H:%M:%S')
                elif status == 'exited' and current_entry:
                    duration = dt_local - current_entry
                    total_duration += duration
                    last_exit_str = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                    current_entry = None

            if total_duration.total_seconds() > 0:
                total_seconds = int(total_duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                duration_str = f"{hours}h {minutes}m {seconds}s"
            else:
                duration_str = 'N/A'

            if total_duration.total_seconds() > 0:
                if total_duration.total_seconds() >= time_limit.total_seconds():
                    present_status = "Present"
                else:
                    present_status = "Half Day"
            else:
                present_status = "N/A"

            summary.append({
                'user_name': user_name,
                'first_entry': first_entry_str,
                'last_exit': last_exit_str,
                'duration': duration_str,
                'present': present_status
            })

        c.execute("SELECT COUNT(*) FROM users WHERE user_type='employee'")
        total_employees = c.fetchone()[0]
        present_today = len(summary)

        return render_template("org-templates/org-dashboard.html", 
                               summary=summary, 
                               total_employees=total_employees, 
                               present_today=present_today,
                               date=selected_date,
                               selected_date=selected_date_str,
                               present=present_status)


@app.route('/link-camera', methods=["GET", "POST"])
@login_required
def linkCamera():
    global camera, stream_thread, streaming

    camera_url = None

    if request.method == "POST":
        # Get the camera URL from the form
        camera_url = request.form.get("camera-url")

        # Start the OpenCV stream in a separate thread
        if not streaming:
            try:
                stream_thread = threading.Thread(target=open_camera_stream, args=(camera_url,))
                stream_thread.start()
                flash("Stream started successfully.", "success")
            except Exception as e:
                flash(f"Error starting stream: {e}", "error")
        else:
            flash("Stream is already running.", "warning")

    return render_template("org-templates/link-camera.html", streaming=streaming, camera_url=camera_url)


@app.route('/kill-stream', methods=["POST"])
def kill_stream():
    global streaming
    if streaming:
        streaming = False  # Stop the OpenCV stream
    return redirect(url_for('linkCamera'))

@app.route('/employee-attendance')
@login_required
def employeeAttendance():
    db, c = get_db()

    # Get selected date
    date_str = request.args.get('date')
    if not date_str:
        selected_date = datetime.datetime.now().date()
    else:
        selected_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    start_datetime = f"{selected_date} 00:00:00"
    end_datetime = f"{selected_date} 23:59:59"

    c.execute("""
        SELECT u.user_name, a.timestamp, a.status
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        WHERE a.timestamp BETWEEN ? AND ?
        ORDER BY u.user_name, a.timestamp
    """, (start_datetime, end_datetime))

    records = c.fetchall()

    LOCAL_TZ = ZoneInfo('Asia/Kolkata')
    user_logs = {}

    # Collect records
    for user_name, timestamp, status in records:
        if user_name not in user_logs:
            user_logs[user_name] = []
        user_logs[user_name].append((timestamp, status))

    attendance_summary = []

    for user_name, logs in user_logs.items():
        sessions = []
        current_entry = None

        for timestamp, status in logs:
            dt_utc = datetime.datetime.fromisoformat(timestamp).replace(tzinfo=ZoneInfo('UTC'))
            dt_local = dt_utc.astimezone(LOCAL_TZ)
            if status == 'entered':
                current_entry = dt_local
            elif status == 'exited' and current_entry:
                duration = dt_local - current_entry
                sessions.append({
                    'entry': current_entry.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit': dt_local.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': str(duration)
                })
                current_entry = None

        attendance_summary.append({
            'user_name': user_name,
            'date': selected_date.strftime('%Y-%m-%d'),
            'logs': sessions
        })

    return render_template(
        "org-templates/employee-attendance.html",
        attendance_summary=attendance_summary,
        selected_date=selected_date.strftime('%Y-%m-%d')
    )

'''@app.route('/employee-attendance')
@login_required
def employeeAttendance():
    db, c = get_db()

    # Get selected date from query params
    date_str = request.args.get('date')
    if not date_str:
        selected_date = datetime.datetime.now().date()
    else:
        selected_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    start_datetime = f"{selected_date} 00:00:00"
    end_datetime = f"{selected_date} 23:59:59"

    # Step 1: Fetch ALL records for the day (we will pair them manually)
    c.execute("""
        SELECT u.user_name, a.timestamp, a.status
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        WHERE a.timestamp BETWEEN ? AND ?
        ORDER BY u.user_name, a.timestamp ASC
    """, (start_datetime, end_datetime))

    records = c.fetchall()

    # Step 2: Prepare
    LOCAL_TZ = ZoneInfo('Asia/Kolkata')
    user_sessions = {}

    for user_name, timestamp, status in records:
        if user_name not in user_sessions:
            user_sessions[user_name] = []
        user_sessions[user_name].append((timestamp, status))

    # Step 3: Process
    attendance_summary = []

    for user_name, sessions in user_sessions.items():
        total_duration = datetime.timedelta()
        first_entry_str = 'N/A'
        last_exit_str = 'N/A'
        current_entry = None

        for timestamp, status in sessions:
            dt_utc = datetime.datetime.fromisoformat(timestamp).replace(tzinfo=ZoneInfo('UTC'))
            dt_local = dt_utc.astimezone(LOCAL_TZ)

            if status == 'entered':
                if not current_entry:
                    current_entry = dt_local
                    if first_entry_str == 'N/A':
                        first_entry_str = current_entry.strftime('%Y-%m-%d %H:%M:%S')

            elif status == 'exited' and current_entry:
                duration = dt_local - current_entry
                total_duration += duration
                last_exit_str = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                current_entry = None

        if total_duration.total_seconds() > 0:
            total_seconds = int(total_duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            duration_str = f"{hours}h {minutes}m {seconds}s"
        else:
            duration_str = 'N/A'

        attendance_summary.append({
            'user_name': user_name,
            'date': selected_date.strftime('%Y-%m-%d'),
            'first_entry': first_entry_str,
            'last_exit': last_exit_str,
            'total_duration': duration_str
        })

    return render_template(
        "org-templates/employee-attendance.html",
        attendance_summary=attendance_summary,
        selected_date=selected_date.strftime('%Y-%m-%d')
    )
'''

@app.route('/my-attendance')
@login_required
def myAttendance():
    db, c = get_db()
    user_id = session['user_id']

    # Get selected date from query string
    selected_date_str = request.args.get('date')
    
    if selected_date_str:
        selected_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").date()
    else:
        selected_date = datetime.date.today()

    start_datetime = datetime.datetime.combine(selected_date, datetime.datetime.min.time())
    end_datetime = datetime.datetime.combine(selected_date, datetime.datetime.max.time())

    # Fetch all attendance records for the user
    c.execute("""
        SELECT timestamp, status
        FROM attendance
        WHERE user_id = ?
        AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, (user_id, start_datetime, end_datetime))
    
    records = c.fetchall()

    # Prepare attendance_summary as list of individual entry-exit pairs
    attendance_summary = []
    current_entry = None

    # Timezones
    utc = pytz.utc
    local_tz = pytz.timezone('Asia/Kolkata')  # Change if needed

    for timestamp_str, status in records:
        dt_utc = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        dt_local = utc.localize(dt_utc).astimezone(local_tz)

        if status == 'entered':
            current_entry = dt_local

        elif status == 'exited' and current_entry:
            duration = dt_local - current_entry

            attendance_summary.append({
                'date': current_entry.strftime('%Y-%m-%d'),
                'first_entry': current_entry.strftime('%I:%M %p'),
                'last_exit': dt_local.strftime('%I:%M %p'),
                'duration': str(duration)
            })

            current_entry = None

    # If there is an entry without exit (still in work)
    if current_entry:
        attendance_summary.append({
            'date': current_entry.strftime('%Y-%m-%d'),
            'first_entry': current_entry.strftime('%I:%M %p'),
            'last_exit': 'In Work',
            'duration': 'In Work'
        })

    return render_template(
        "user-templates/my-attendance.html",
        attendance_summary=attendance_summary, 
        selected_date=selected_date
    )


@app.route('/profile')
@login_required
def orgProfile():
    if session['user_type'] == 'employee':
        db, c = get_db()
        images = c.execute("SELECT image_path, id FROM userImages WHERE user_id = ?", (session['user_id'],)).fetchall()
        
        trained_images = c.execute(""
        "SELECT ui.id "
        "FROM trainingStatus ts "
        "JOIN userImages ui "
        "ON ts.image_id = ui.id "
        "WHERE ui.user_id = ? "
        "AND ts.is_trained = 1", 
        (session['user_id'],)).fetchall()

        trained_image_ids = [img[0] for img in trained_images]

        image_length = len(images)   
        trained_image_length = len(trained_images)

        return render_template(
                    "user-templates/profile.html",
                    images=images,
                    image_length=image_length,
                    trained_images=trained_image_ids,
                    trained_image_length=trained_image_length,
                )

@app.route('/addImages', methods=['POST'])
@login_required
def addImages():
    # Validate 'name' input
    name = request.form.get('name')
    if not name or name.strip() == '':
        flash('Name is required')
        return redirect(url_for('orgProfile'))
    
    if 'images' not in request.files:
        flash('No files part in the request')
        return redirect(url_for('orgProfile'))
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        flash('No files selected')
        return redirect(url_for('orgProfile'))

    if len(files) > 9:
        flash('You can only upload up to 9 images')
        return redirect(url_for('orgProfile'))
    
    user_id = session['user_id']
    db, c = get_db()

    # --- Update user's name in the users table ---
    c.execute('''
        UPDATE users
        SET user_name = ?
        WHERE id = ?
    ''', (name.strip(), user_id))

    session['user_name'] = name.strip()


    c.execute("SELECT COUNT(*) FROM userImages WHERE user_id = ?", (user_id,))
    existing_images_count = c.fetchone()[0]

    if existing_images_count >= 9:
        flash('You already have 9 images uploaded')
        return redirect(url_for('orgProfile'))

    uploaded_count = 0  # <-- Track successful uploads

    for file in files:
        if file and allowed_file(file.filename) and file.filename != '':
            if existing_images_count < 9:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                normalized_path = file_path.replace("\\", "/")
                
                c.execute('''
                    INSERT INTO userImages (user_id, image_path) 
                    VALUES (?, ?)
                ''', (user_id, normalized_path))
                
                existing_images_count += 1
                uploaded_count += 1
            else:
                flash('Upload limit reached. Some images were not uploaded.')
                break
        else:
            if file.filename != '':
                flash(f'Invalid file type: {file.filename}')

    db.commit()
    db.close()

    if uploaded_count > 0:
        flash('Images uploaded successfully')
    else:
        flash('No valid images were uploaded')

    return redirect(url_for('orgProfile'))


@app.route('/reset-user-images', methods=['POST'])
@login_required
def reset_user_images():
    db, cursor = get_db()
    user_id = session['user_id']

    # 🔍 Step 1: Get all image IDs & file paths for the logged-in user
    cursor.execute("SELECT id, image_path FROM userImages WHERE user_id = ?", (user_id,))
    user_images = cursor.fetchall()

    if not user_images:
        return "No images found to reset.", 200

    # 🗑 Step 2: Delete only the logged-in user's images from userImages
    cursor.execute("DELETE FROM userImages WHERE user_id = ?", (user_id,))
    db.commit()

    # 🗑 Step 3: Delete trainingStatus records related to those images
    image_ids = [str(img[0]) for img in user_images]
    cursor.execute("DELETE FROM trainingStatus WHERE image_id IN ({})".format(
        ",".join(["?"] * len(image_ids))
    ), image_ids)
    db.commit()

    # 🧹 Step 4: Delete only the logged-in user's uploaded images from 'static/uploads'
    for _, image_path in user_images:
        file_path = os.path.join(UPLOAD_FOLDER, image_path)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')

    flash("Your images and training data have been reset.", "success")
    return redirect(request.referrer)


@app.route('/trainImages', methods=['POST'])
@login_required
def trainImages():
    user_id = session['user_id']
    user_name = session['user_name']
    db, c = get_db()
    
    c.execute("SELECT id, image_path FROM userImages WHERE user_id = ?", (user_id,))
    user_images = c.fetchall()
    
    if len(user_images) < 9:
        flash('You need 9 images uploaded to start training')
        return redirect(url_for('orgProfile'))
    
    upload_folder_location = app.config['UPLOAD_FOLDER']
    trained_count = 0
    
    for image_id, image_filename in user_images:
        image_path = os.path.join(upload_folder_location, image_filename)
        success, message = train_image(image_filename, user_name, user_id)
        
        if success:
            trained_count += 1
            c.execute('''
                INSERT OR REPLACE INTO trainingStatus (image_id, is_trained, trained_at)
                VALUES (?, ?, ?)
            ''', (image_id, True, datetime.datetime.now()))
        else:
            flash(f'Error training image {image_filename}: {message}')
    
    if trained_count == len(user_images):
        flash('Training completed successfully')
    else:
        flash(f'Training completed with some errors. {trained_count}/{len(user_images)} trained successfully.')
    
    db.commit()
    db.close()
    return redirect(url_for('orgProfile'))


@app.route('/admin_clear_all_attendance')
def admin_clear_all_attendance():
    # Add admin check here
    db, cursor = get_db()
    cursor.execute("DELETE FROM attendance")
    db.commit()
    return "All attendance records cleared.", 200