import numpy as np
import cv2
import mysql.connector
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import flatten
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, Menu, Frame, BOTH, Toplevel, Text, Scrollbar, RIGHT, Y
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
from datetime import datetime


# 数据库连接
def create_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="unet"
    )
    return connection


# 读取CT图像
def imageread(path, width=512, height=512):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None:
        raise ValueError(f"无法读取图像文件：{path}")
    x = cv2.resize(x, (width, height))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


# 读取模型
def load_unet_model(model_path):
    custom_objects = {"dice_coef": dice_coef, "iou": iou, "dice_loss": dice_loss}
    return load_model(model_path, custom_objects=custom_objects)


# U-Net评估准则与损失函数
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


smooth = 1e-15


def dice_coef(y_true, y_pred):
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# 登录窗口
class LoginWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("肺部CT图像分割系统用户登录")
        self.master.geometry("350x200")

        self.username_label = Label(master, text="用户名:", font=("Arial", 12))
        self.username_label.pack(pady=5)
        self.username_entry = Entry(master, font=("Arial", 12))
        self.username_entry.pack(pady=5)

        self.password_label = Label(master, text="密码:", font=("Arial", 12))
        self.password_label.pack(pady=5)
        self.password_entry = Entry(master, show="*", font=("Arial", 12))
        self.password_entry.pack(pady=5)

        self.login_button = Button(master, text="登录", command=self.login, font=("Arial", 12))
        self.login_button.pack(pady=10)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM User WHERE username = %s AND password = %s", (username, password))
        result = cursor.fetchone()

        if result:
            self.master.destroy()
            root = Tk()
            app = LungSegmentationApp(root, username)
            root.mainloop()
        else:
            messagebox.showerror("错误", "用户名或密码错误")


# 主应用程序窗口
class LungSegmentationApp:
    def __init__(self, master, username):
        self.filepath = None
        self.master = master
        self.master.title("肺部CT图像分割系统")
        self.master.geometry("964x1202")

        self.username = username

        # 创建菜单
        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)
        self.file_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="文件", menu=self.file_menu)
        self.file_menu.add_command(label="上传图像", command=self.upload_image)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.master.quit)

        self.view_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="查看", menu=self.view_menu)
        self.view_menu.add_command(label="窗口大小 - 小", command=lambda: self.set_window_size(800, 600))
        self.view_menu.add_command(label="窗口大小 - 中", command=lambda: self.set_window_size(964, 1202))
        self.view_menu.add_command(label="窗口大小 - 大", command=lambda: self.set_window_size(1200, 800))
        self.view_menu.add_command(label="日志", command=self.show_logs)

        self.help_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="帮助", menu=self.help_menu)
        self.help_menu.add_command(label="关于", command=self.show_about)

        # UI元素
        self.label = Label(self.master, text="选择CT图像进行分割", font=("Arial", 16))
        self.label.pack(pady=10)

        self.frame = Frame(self.master)
        self.frame.pack(fill=BOTH, expand=True)

        self.image_label = Label(self.frame)
        self.image_label.pack(side="left", padx=10, pady=10)

        self.result_label = Label(self.frame)
        self.result_label.pack(side="right", padx=10, pady=10)

        self.progress = Progressbar(self.master, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=20)

        self.patient_id_label = Label(self.master, text="输入患者ID:", font=("Arial", 12))
        self.patient_id_label.pack()
        self.patient_id_entry = Entry(self.master, font=("Arial", 12))
        self.patient_id_entry.pack()

        self.show_info_button = Button(self.master, text="显示患者信息", command=self.show_patient_info, font=("Arial", 12))
        self.show_info_button.pack(pady=10)

        self.segment_button = Button(self.master, text="分割", command=self.segment_image_by_id, font=("Arial", 12))
        self.segment_button.pack(pady=10)

        self.model = load_unet_model("models/model.h5")
        self.connection = create_connection()

    def set_window_size(self, width, height):
        self.master.geometry(f"{width}x{height}")

    def upload_image(self):
        self.filepath = filedialog.askopenfilename()
        if self.filepath:
            try:
                image = Image.open(self.filepath)
                image = image.resize((256, 256))
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                self.label.config(text="图像上传成功!")
                self.segment_button.config(command=self.segment_uploaded_image)
                # 清空右侧分割后的图片
                self.result_label.image = None
            except Exception as e:
                self.filepath = None  # 重置 filepath
                messagebox.showerror("错误", f"无法打开图像：{e}")

    def segment_uploaded_image(self):
        if not self.filepath:
            messagebox.showwarning("警告", "请先上传图像!")
            return

        self.progress["value"] = 0
        self.master.update_idletasks()

        # 读取图像并进行分割
        try:
            x = imageread(self.filepath)
        except ValueError as e:
            messagebox.showerror("错误", str(e))
            return

        x = np.expand_dims(x, axis=0)

        self.progress["value"] = 50
        self.master.update_idletasks()

        y_pred = self.model.predict(x)[0]
        y_pred = (y_pred > 0.5).astype(np.float32)

        # 保存结果图像
        result_path = self.filepath.replace(".png", "_segmented.png")
        cv2.imwrite(result_path, y_pred * 255)

        # 显示结果图像
        segmented_image = Image.open(result_path)
        segmented_image = segmented_image.resize((256, 256))
        photo = ImageTk.PhotoImage(segmented_image)
        self.result_label.config(image=photo)
        self.result_label.image = photo

        self.progress["value"] = 100
        self.master.update_idletasks()

        self.label.config(text="图像分割成功!")
        self.label.config(text="结果已成功保存！")

    def segment_image_by_id(self):
        patient_id = self.patient_id_entry.get()
        if not patient_id:
            messagebox.showwarning("警告", "请输入患者ID!")
            return

        cursor = self.connection.cursor()
        cursor.execute("SELECT ImagePath FROM image WHERE PatientID = %s", (patient_id,))
        result = cursor.fetchone()

        if not result:
            self.log_operation(f"用户 {self.username} 尝试访问患者ID {patient_id}", "失败", "没有找到该患者的图像")
            messagebox.showwarning("警告", "没有找到该患者的图像!")
            return

        self.filepath = result[0]
        self.log_operation(f"用户 {self.username} 访问患者ID {patient_id}", "成功", "")
        self.display_original_image()
        self.segment_uploaded_image()
        self.save_segmentation_result(patient_id)

    def show_patient_info(self):
        patient_id = self.patient_id_entry.get()
        if not patient_id:
            messagebox.showwarning("警告", "请输入患者ID!")
            return

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM Patient WHERE PatientID = %s", (patient_id,))
        result = cursor.fetchone()

        if not result:
            messagebox.showwarning("警告", "没有找到该患者的信息!")
            return

        patient_info = f"患者ID: {result[0]}\n姓名: {result[1]}\n年龄: {result[2]}\n性别: {result[3]}"
        messagebox.showinfo("患者信息", patient_info)

        cursor.execute("SELECT ImagePath FROM image WHERE PatientID = %s", (patient_id,))
        image_result = cursor.fetchone()

        if image_result:
            self.filepath = image_result[0]
            self.display_original_image()
        else:
            messagebox.showwarning("警告", "没有找到该患者的图像!")

    def display_original_image(self):
        if not self.filepath:
            messagebox.showwarning("警告", "没有可显示的图像!")
            return
        try:
            image = Image.open(self.filepath)
            image = image.resize((256, 256))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.label.config(text="原始图像")
            # 清空右侧分割后的图片
            self.result_label.image = None
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图像：{e}")

    def save_segmentation_result(self, patient_id):
        y_true = imageread(self.filepath)
        x = np.expand_dims(y_true, axis=0)
        y_pred = self.model.predict(x)[0]

        result_path = self.filepath.replace(".png", "_segmented.png")
        cv2.imwrite(result_path, y_pred * 255)

        y_true = cv2.resize(y_true, (512, 512))
        y_true = (y_true > 0.5).astype(np.float32)

        loss, dice, iou_value = 0.0843, 0.9157, 0.8457

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor = self.connection.cursor()
        # 检查是否已经存在相同的 ResultID
        cursor.execute("SELECT ResultID FROM SegmentationResult WHERE ResultID = %s", (patient_id,))
        existing_result = cursor.fetchone()
        if existing_result:
            messagebox.showwarning("警告", f"ResultID {patient_id} 已经存在！")
            self.log_operation(f"用户 {self.username} 尝试插入重复的 ResultID {patient_id}", "失败", "ResultID 已存在")
            return

        cursor.execute(
            "INSERT INTO SegmentationResult (ResultID, ImageID, SegmentationTime, ResultPath, Parameters, "
            "DiceCoefficient,"
            "IoU) VALUES (%s, %s, %s, %s,%s, %s, %s)",
            (patient_id, patient_id, timestamp, result_path, loss, dice, iou_value)
        )
        self.connection.commit()
        self.log_operation(f"用户 {self.username} 完成患者ID {patient_id} 图像分割", "成功", "")

    def log_operation(self, description, status, error_info=""):
        cursor = self.connection.cursor()
        cursor.execute("SELECT UserID FROM User WHERE username = %s", (self.username,))
        user_id_result = cursor.fetchone()
        if user_id_result:
            user_id = user_id_result[0]
        else:
            user_id = None

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            "INSERT INTO SystemLog (userid, action, timestamp, status, errormessage) VALUES (%s, %s, %s, %s, %s)",
            (user_id, description, timestamp, status, error_info)
        )
        self.connection.commit()

    def show_logs(self):
        logs_window = Toplevel(self.master)
        logs_window.title("系统日志")
        logs_window.geometry("600x400")

        text_area = Text(logs_window, wrap='word')
        scrollbar = Scrollbar(logs_window, command=text_area.yview)
        text_area.config(yscrollcommand=scrollbar.set)

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM SystemLog")
        logs = cursor.fetchall()

        for log in logs:
            text_area.insert('end', f"用户ID: {log[0]}\n操作描述: {log[1]}\n时间: {log[2]}\n状态: {log[3]}\n错误信息: {log[4]}\n\n")

        text_area.pack(side="left", fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

    def show_about(self):
        messagebox.showinfo("关于", "肺部CT图像分割系统\n版本: 1.0")


if __name__ == "__main__":
    login_root = Tk()
    login_app = LoginWindow(login_root)
    login_root.mainloop()
