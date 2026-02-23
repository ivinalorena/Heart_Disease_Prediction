import sqlite3

DB_PATH = "pacientes_HD.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pacientes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        idade INTEGER NOT NULL CHECK (idade >= 0),
        sexo TEXT NOT NULL,
        resultado TEXT NOT NULL,
        criado_em DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    return conn

# Inicializa o banco ao importar/executar este arquivo
conn = init_db()
cursor = conn.cursor()
