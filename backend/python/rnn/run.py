import threading
import time
from app import create_app

app = create_app()

def periodic_task():
    while True:
        print("Task dijalankan setiap 1 menit.")
        # Lakukan tugas di sini (misal update database, baca sensor, dll)
        time.sleep(300)

# Jalankan tugas background setelah app siap
threading.Thread(target=periodic_task, daemon=True).start()

if __name__ == '__main__':
    app.socketio.run(app.flask_app, debug=False, host='0.0.0.0', 
                     port=5000, allow_unsafe_werkzeug=True)
