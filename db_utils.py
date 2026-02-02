import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            employee_id TEXT,
            department TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_employee(name, employee_id=None, department=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO employees (name, employee_id, department) 
            VALUES (?, ?, ?)
        ''', (name, employee_id, department))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # If employee already exists, we might want to update or just skip
        return False
    finally:
        conn.close()

def get_employees():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, employee_id, department, created_at FROM employees ORDER BY name ASC')
    employees = cursor.fetchall()
    conn.close()
    return [{"name": row[0], "employee_id": row[1], "department": row[2], "created_at": row[3]} for row in employees]

def log_attendance(name):
    if name == "Unknown":
        return False
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Optional: Prevent duplicate logs within a short timeframe (e.g., 5 minutes)
    cursor.execute('''
        SELECT timestamp FROM attendance_logs 
        WHERE name = ? 
        ORDER BY timestamp DESC LIMIT 1
    ''', (name,))
    last_log = cursor.fetchone()
    
    if last_log:
        last_time = datetime.strptime(last_log[0], '%Y-%m-%d %H:%M:%S')
        diff = datetime.now() - last_time
        if diff.total_seconds() < 300: # 5 minutes
            conn.close()
            return False

    cursor.execute('INSERT INTO attendance_logs (name) VALUES (?)', (name,))
    conn.commit()
    conn.close()
    return True

def get_logs(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, timestamp FROM attendance_logs ORDER BY timestamp DESC LIMIT ?', (limit,))
    logs = cursor.fetchall()
    conn.close()
    return [{"name": row[0], "timestamp": row[1]} for row in logs]

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
