import sqlite3

# Connect to the database
conn = sqlite3.connect("transactions.db")
cursor = conn.cursor()

# Add the missing column 'reason' if it does not exist
try:
    cursor.execute("ALTER TABLE fraud_reports ADD COLUMN reason TEXT;")
    print("✅ Column 'reason' added successfully!")
except sqlite3.OperationalError:
    print("⚠️ Column 'reason' already exists!")

# Commit and close
conn.commit()
conn.close()
