import string
from typing import List
import service.databse as db;
import models.ClassPaciente as paciente;
import sqlite3
import csv

def db_insert(nome, idade, sexo, resultado):
    db.cursor.execute(
        """
        INSERT INTO pacientes (nome, idade, sexo, resultado)
        VALUES (?, ?, ?, ?)
        """,
        (nome, idade, sexo, resultado)
    )
    db.conn.commit()

def excluir(id_paciente):
    sql = "DELETE FROM pacientes WHERE id = ?"
    db.cursor.execute(sql, (id_paciente,))
    db.conn.commit()

def selecionar_todos():
    db.cursor.execute("SELECT id, nome, idade, sexo, resultado, criado_em FROM pacientes")
    pacientes_list = []
    for row in db.cursor.fetchall():
        # ajuste o construtor conforme sua classe real
        pacientes_list.append(paciente.Paciente(row[0], row[1], row[2], row[3], row[4], row[5]))
    return pacientes_list
