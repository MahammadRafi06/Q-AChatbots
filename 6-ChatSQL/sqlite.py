import sqlite3
connection = sqlite3.connect("student.db")
cursor = connection.cursor()

table_info = """ create table STUDENT(
                   NAME VARCHAR(30),
                   CLASS VARCHAR(20),
                   SECTION VARCHAR(25),
                   MARKS INT  
)
"""
cursor.execute(table_info)

cursor.execute("""INSERT INTO STUDENT values('Mahammad','Data Science','A','90')""")
cursor.execute("""INSERT INTO STUDENT values('Shuaib','Data Science','B','100')""")
cursor.execute("""INSERT INTO STUDENT values('Idrees','DevOps','A','96')""")
cursor.execute("""INSERT INTO STUDENT values('Tasneem','DevOps','A','99')""")

print("The inserted records are")
data= cursor.execute("""SELECT * FROM STUDENT""")

for row in data:
    print(row)
connection.commit()
connection.close()