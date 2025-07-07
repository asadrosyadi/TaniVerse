import RPi.GPIO as GPIO
import time
import serial
import pymysql
import atexit

# ==================== Konfigurasi Hardware ====================
SERVO_PIN = 18
TF_LUNA_PORT = '/dev/ttyAMA0'
TF_LUNA_BAUDRATE = 115200

# ==================== Konfigurasi Database ====================
DB_HOST = 'localhost'
DB_USER = 'pi'
DB_PASS = 'Petaniasik123#$'
DB_NAME = 'padi_sawah'
IOT_ID = "jTZids5M"  # Ubah sesuai ID unik perangkat Anda

# ==================== Parameter Sistem ====================
MOVEMENT_THRESHOLD = 0.2
DETECTION_INTERVAL = 0.5
DIRECTION_CHANGE_INTERVAL = 3

# ==================== Inisialisasi GPIO Servo ====================
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

# ==================== Variabel Global ====================
PREV_DISTANCE = 0

# ==================== Fungsi Servo ====================
def set_servo_speed(speed):
    """Atur kecepatan servo (-100 ke 100)"""
    speed = max(-100, min(100, speed))
    duty = 7.5 + (speed * 2.5 / 100)
    pwm.ChangeDutyCycle(duty)

def stop_servo():
    pwm.ChangeDutyCycle(7.5)
    time.sleep(0.5)
    pwm.stop()
    GPIO.cleanup()

# ==================== Deteksi Gerakan ====================
def detect_movement(current_dist):
    global PREV_DISTANCE
    if PREV_DISTANCE == 0:
        PREV_DISTANCE = current_dist
        return False
    movement = abs(current_dist - PREV_DISTANCE) > MOVEMENT_THRESHOLD
    PREV_DISTANCE = current_dist
    return movement

# ==================== Inisialisasi TF-Luna ====================
try:
    tf_luna = serial.Serial(
        port=TF_LUNA_PORT,
        baudrate=TF_LUNA_BAUDRATE,
        timeout=1,
        bytesize=8,
        parity='N',
        stopbits=1
    )
    time.sleep(1)
except Exception as e:
    print(f"ERROR koneksi TF-Luna: {str(e)}")
    exit(1)

# ==================== Inisialisasi MySQL ====================
try:
    db_connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    with db_connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lidars (
                id BIGINT(20) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                iot_id TEXT NOT NULL,
                distance DECIMAL(5,2) NOT NULL,
                movement_detected INT(11) NOT NULL,
                servo_position TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        # Insert baris baru jika belum ada
        cursor.execute("SELECT COUNT(*) FROM lidars WHERE iot_id = %s", (IOT_ID,))
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.execute("INSERT INTO lidars (iot_id, distance, movement_detected, servo_position) VALUES (%s, 0, 0, 'N/A')", (IOT_ID,))
        db_connection.commit()
except Exception as e:
    print(f"GAGAL koneksi MySQL: {str(e)}")
    exit(1)

# ==================== Fungsi Bersih-bersih ====================
def cleanup():
    stop_servo()
    if 'tf_luna' in globals() and tf_luna.is_open:
        tf_luna.close()
    if 'db_connection' in globals() and db_connection.open:
        db_connection.close()

atexit.register(cleanup)

# ==================== Fungsi Pembacaan Sensor ====================
def get_distance():
    try:
        tf_luna.reset_input_buffer()
        tf_luna.write(b'\x5A\x04\x11\x6F')
        time.sleep(0.05)
        if tf_luna.in_waiting >= 9:
            data = tf_luna.read(9)
            if data[0] == 0x59 and data[1] == 0x59:
                distance = data[2] + data[3] * 256
                strength = data[4] + data[5] * 256
                if strength > 100 and 30 <= distance <= 800:
                    return distance / 100.0
    except Exception as e:
        print(f"Error baca sensor: {str(e)}")
    return None

# ==================== Fungsi Update Database ====================
def update_db(distance, movement, direction):
    try:
        with db_connection.cursor() as cursor:
            sql = """
                UPDATE lidars 
                SET distance = %s, 
                    movement_detected = %s, 
                    servo_position = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE iot_id = %s
            """
            cursor.execute(sql, (distance, int(movement), direction, IOT_ID))
        db_connection.commit()
    except Exception as e:
        print(f"Error update database: {str(e)}")

# ==================== Fungsi Utama ====================
def main():
    global PREV_DISTANCE
    try:
        direction = "CW"
        set_servo_speed(60)
        last_detection = time.time()
        last_direction_change = time.time()

        while True:
            current_time = time.time()

            # Baca sensor jarak
            if current_time - last_detection >= DETECTION_INTERVAL:
                distance = get_distance()
                if distance is not None:
                    movement = detect_movement(distance)
                    print(f"Jarak: {distance:.2f}m | Gerakan: {movement} | Arah: {direction}")
                    update_db(distance, movement, direction)
                last_detection = current_time

            # Ganti arah putar servo
            if current_time - last_direction_change > DIRECTION_CHANGE_INTERVAL:
                direction = "CCW" if direction == "CW" else "CW"
                speed = -60 if direction == "CCW" else 60
                set_servo_speed(speed)
                last_direction_change = current_time

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nDiterima sinyal berhenti dari keyboard")
    except Exception as e:
        print(f"\nERROR: {str(e)}")

# ==================== Jalankan Program ====================
if __name__ == "__main__":
    main()
