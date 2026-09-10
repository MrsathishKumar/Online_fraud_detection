import sqlite3

def create_table():
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()

    # Table for transactions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amount REAL,
            phone_number TEXT,
            location TEXT,
            fraud_probability REAL,
            prediction TEXT
        )
    """)

    # Table for reported fraud cases
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reported_fraud (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amount REAL,
            phone_number TEXT,
            location TEXT,
            fraud_probability REAL,
            status TEXT DEFAULT 'Under Investigation'
        )
    """)

    conn.commit()
    conn.close()

# Initialize database
create_table()
print("✅ Database initialized successfully.")
