import mysql.connector


# 创建数据库连接
def create_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="unet"
    )
    return connection


# 插入预先数据
def insert_data(connection):
    cursor = connection.cursor()

    # 插入用户表数据
    users = [
        (1, "admin", "123", "管理员"),
        (2, "doctor1", "456", "普通用户"),
        (3, "doctor2", "789", "普通用户")
    ]
    cursor.executemany("INSERT INTO User (UserID, Username, Password, Role) VALUES (%s, %s, %s, %s)", users)

    # 插入患者表数据
    patients = [
        (1,"John Doe", 45, "Male", "123-456-7890", "History of asthma"),
        (2,"Jane Smith", 37, "Female", "987-654-3210", "No significant history")
    ]
    cursor.executemany(
        "INSERT INTO Patient (PatientID,Name, Age, Gender, ContactInfo, MedicalHistory) VALUES (%s, %s, %s, %s, %s, %s)",
        patients)

    # 插入图像表数据
    images = [
        (1, 1, "2024-05-18 10:00:00", r"E:\PycharmProjects\Unet\data\CXR_png\MCUCXR_0001_0.png", "CT Scan - John Doe"),
        (2, 2, "2024-05-18 11:00:00", r"E:\PycharmProjects\Unet\data\CXR_png\MCUCXR_0002_0.png", "CT Scan - Jane Smith")
    ]
    cursor.executemany("INSERT INTO Image (ImageID, PatientID, UploadTime, ImagePath, Description) VALUES (%s, %s, %s, "
                       "%s, %s)",
                       images)
    # 提交事务
    connection.commit()


# 主函数
if __name__ == "__main__":
    connection = create_connection()
    insert_data(connection)
    connection.close()
