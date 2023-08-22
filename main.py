from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from datetime import date
import cv2, os,numpy
import numpy as np           
from sklearn.neighbors import KDTree
from sample_onnx import main,get_args
import time
ui,_=loadUiType('face_recongtion.ui')
yunet = main()
args = get_args()
model_path = args.model
input_shape = tuple(map(int, args.input_shape.split(',')))
score_th = args.score_th
nms_th = args.nms_th
topk = args.topk
keep_topk = args.keep_topk
print(ui)
class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)# tap hiển thị trong tiên trong tap widget của app 
        self.LOGIN.clicked.connect(self.login)
        self.CLOSE.clicked.connect(self.close)
        self.LOGOUT.clicked.connect(self.logout)
        self.train_user.clicked.connect(self.show_training)
        self.attendt_entry.clicked.connect(self.show_attendent)
        self.reports_tab2.clicked.connect(self.show_reportstab2)
        self.goback_training.clicked.connect(self.show_mainfrom)
        self.face_recong_back.clicked.connect(self.show_training)
        self.reports_back.clicked.connect(self.show_face_recongize)
        self.train.clicked.connect(self.start_training)
        self.Record.clicked.connect(lambda: self.record_attendance(0, "video_output"))
        self.dateEdit.setDate(date.today())
        self.dateEdit.dateChanged.connect(self.show_selected_date_reports)# thay đổi ngày
        self.tabWidget.setStyleSheet("QTabWidget::pane{border:0;}")# xoá đường viền 

        try:
            con = sqlite3.connect("face-reco.db")
            con.execute("CREATE TABLE IF NOT EXISTS attendance(attendanceid INTEGER, name TEXT, attendancedate TEXT)")
            #nếu bảng đã tồn tại thì ko thực thi câu lệnh , gồm 3 cột và kiểu data ở phía sau 
            con.commit()# lưu thay đổi 
            print("Table created successfully")
        except:
            print("error in database!")

        ## login process
    def login(self):
        pw = self.PASSWORD.text()
        if(pw=="114"):
            self.PASSWORD.setText("")
            self.tabWidget.setCurrentIndex(1)# chuyển tab tiếp theo 
        else:
            self.FAILWORD.setText("Invalid password!")
            self.PASSWORD.setText("")
    def logout (self):
        self.tabWidget.setCurrentIndex(0)
    def close(self):
        self.close()
    def show_training(self):
         self.tabWidget.setCurrentIndex(2)
    def show_attendent(self):
         self.tabWidget.setCurrentIndex(3)
    def show_reportstab2(self):
         self.tabWidget.setCurrentIndex(4)
         self.REPORTTAB4.setRowCount(0)# xoá tất cả hàng
         self.REPORTTAB4.clear()# thực hiện xoá
         con = sqlite3.connect("face-reco.db")
         cursor = con.execute("SELECT * FROM attendance")
         result = cursor.fetchall()
         r = 0
         c = 0 
         for row_number, row_data in enumerate(result):
             r += 1
             c = 0 
             for colum_number, data in enumerate(row_data):
                 c+=1
         self.REPORTTAB4.setColumnCount(c)
         for row_number, row_data in enumerate(result):
             self.REPORTTAB4.insertRow(row_number)
             for colum_number, data in enumerate(row_data):
                   self.REPORTTAB4.setItem(row_number,colum_number,QTableWidgetItem(str(data)))# đặt phàn tử data vào vị trí đc chỉ định hàng ? và cột ? 
         self.REPORTTAB4.setHorizontalHeaderLabels(['Id','Name','Date']) # nhãn cho các cột 
         self.REPORTTAB4.setColumnWidth(0,10)# chiều rộng từng cột 
         self.REPORTTAB4.setColumnWidth(1,30)
         self.REPORTTAB4.setColumnWidth(2,90)
         self.REPORTTAB4.verticalHeader().setVisible(False)# ẩn tiêu đề các hàng trong bảng 
    def show_mainfrom(self):
         self.tabWidget.setCurrentIndex(1)
    def show_training(self):
         self.tabWidget.setCurrentIndex(2)
    def show_face_recongize(self):
         self.tabWidget.setCurrentIndex(3)
    # Training process 
    # sử dụng yunet phát hiện mặt và crop nó vào thư mục lưu trữ , có thể dùng cho database hoặc lấy dữ liệu training facenet ,arcface
    def start_training(self):
        datasets = 'datasets'
        sub_data = self.personame_training.text()
        path = os.path.join(datasets,sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
            print("The new directory is created")
            (width,height) = (130,100)
            webcam = cv2.VideoCapture(0)
            count = 1
            while count < int(self.training_capture_counts.text()) + 1:
                start_time = time.time()
                print(count)
                (_,im) = webcam.read()
                gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                faces, landmarks, scores = yunet.inference(gray)
                elapsed_time = time.time() - start_time
                print(elapsed_time)
                image_width, image_height = gray.shape[1], gray.shape[0]
                for bbox, landmark, score in zip(faces, landmarks, scores):
                    if score_th > score:
                        continue
                    # bouding box
                    x1 = int(image_width * (bbox[0] / input_shape[0]))
                    y1 = int(image_height * (bbox[1] / input_shape[1]))
                    x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
                    y2 = int(image_height * (bbox[3] / input_shape[1])) + y1
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face = gray[x1:x2,y1:y2]
                    face_resize = cv2.resize(face,(width,height))
                    cv2.imwrite('%s/%s.png'%(path,count),face_resize)

                count += 1
                cv2.imshow('OpenCV',im)
                key = cv2.waitKey(100)# độ trễ 10s 
                if key == 27:# 27 =esc 
                    break
            webcam.release()
            cv2.destroyAllWindows()  
            path=""
            QMessageBox.information(self,"Attendance System","Training Completed Successfully") # thôg báo  
            self.personame_training.setText("")
            self.training_capture_counts.setText("10")
    
     ### RECORD ATTENDANCE ###
    def record_attendance(self,source,output_directory):
        self.Record.setText("Process started.. Waiting..")        
        datasets = 'datasets'
        (images,labels,names,id) =([],[],{},0)# id = chỉ số person
        for(subdirs,dirs,files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets,subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + "/" + filename
                    label = id
                    images.append(cv2.imread(path,0))
                    labels.append(int(label))
                id += 1
        (images,labels) = [numpy.array(lis) for lis in [images,labels]]# convert array
        print(images,labels)
        (width, height) = (130,100)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images,labels)   
        webcam = cv2.VideoCapture(source)
        cnt=0
        # tạo đối tượng
        output_path = os.path.join(output_directory, 'output.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(output_path, fourcc, 15.0, (640, 480))
        while True:
            (ret,im) = webcam.read()
            if not ret or ret is None:
                break
            start_time = time.time()
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)
            faces, landmarks, scores = yunet.inference(gray)
            #lặp qua các khuôn mặt đc phát hiện và so sánh với mặt trong database 
            elapsed_time = time.time() - start_time
            print(elapsed_time)
            image_width, image_height = gray.shape[1], gray.shape[0]
            for bbox, landmark, score in zip(faces, landmarks, scores):
                if score_th > score:
                    continue
                # bouding box
                x1 = int(image_width * (bbox[0] / input_shape[0]))
                y1 = int(image_height * (bbox[1] / input_shape[1]))
                x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
                y2 = int(image_height * (bbox[3] / input_shape[1])) + y1
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face = gray[x1:x2,y1:y2]
                face_resize = cv2.resize(face,(width,height))
                prediction = model.predict(face_resize)
                # land mark
                for _, landmark_point in enumerate(landmark):
                    x = int(image_width * (landmark_point[0] / input_shape[0]))
                    y = int(image_height * (landmark_point[1] / input_shape[1]))
                    cv2.circle(im, (x, y), 2, (0, 255, 0), 2)
                if(prediction[1]<800):# khuôn mặt phù hợp 
                    # cv2.putText(im,'%s'%(names[prediction[0]]),(x1-10,x2-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
                    cv2.putText(im,'%s-%.0f'%(names[prediction[0]],prediction[1]),(x1-10,y1-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
                    print(names[prediction[0]])# lable
                    self.Record.setText("Dected face " + names[prediction[0]])  # show form       
                    attendanceid =0
                    available = False
                    try:
                        connection = sqlite3.connect("face-reco.db")
                        cursor = connection.execute("SELECT MAX(attendanceid) from attendance")
                        result = cursor.fetchall()# lấy tất cả hàng thoả mãn điều kiện truy vấn 
                        if result:# nếu có rồi thêm tăng lên 1 hàng nữa vào sau cùng  , còn chưa có thì id của nó bắt đầu từ 1 
                            for maxid in result:
                                attendanceid = int(maxid[0])+1
                    except:
                        attendanceid=1
                    print(attendanceid)    

                    try:
                        con = sqlite3.connect("face-reco.db")
                        cursor = con.execute("SELECT * FROM attendance WHERE name='"+ str(names[prediction[0]]) +"' and attendancedate = '"+ str(date.today()) +"'")
                        # cursor = con.execute("SELECT * FROM attendance WHERE name='"+ str(prediction_name) +"' and attendancedate = '"+ str(date.today()) +"'")
                        result = cursor.fetchall()
                        if result:
                            available=True
                        if(available==False):
                            con.execute("INSERT INTO attendance VALUES("+ str(attendanceid) +",'"+ str(names[prediction[0]]) +"','"+ str(date.today()) +"')")
                            # con.execute("INSERT INTO attendance VALUES("+ str(attendanceid) +",'"+ str(prediction_name) +"','"+ str(date.today()) +"')")
                            con.commit()   
                    except:
                        print("Error in database insert")
                    print("Attendance Registered successfully")
                    self.Record.setText("Attence entered for " + names[prediction[0]])       
                    # self.Record.setText("Attence entered for " +prediction_name)        
                    cnt=0
                    if source!=0 :
                        output_video.write(im)
                    else:
                        continue
                else:
                    cnt+=1# tăng lên 1 nếu khuôn mặt không đc nhận dạng đúng 
                    cv2.putText(im,'UnKnown',(x1-10,y1-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
                    print("Unknown person")
                    self.Record.setText("Unknown Person ")        
                    cv2.imwrite('unKnown.jpg',im)
                    cnt=0
            cv2.imshow("Face Recognition",im)
            # key = cv2.waitKey(10)
            # if key==27:
            #     break
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                    break
            elif key == ord('s'):
                    ref_dir = "./image_process/"
                    save_path = "img_crop.jpg"
                    save_path = os.path.join(ref_dir,save_path)
                    cv2.imwrite(save_path,im)
                    print("An image is saved to ",save_path)
        webcam.release()
        output_video.release()
        cv2.destroyAllWindows()  
 ### SHOW SELECTED DATE REPORTS ###
    def show_selected_date_reports(self):
        self.REPORTTAB4.setRowCount(0)
        self.REPORTTAB4.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance WHERE attendancedate = '"+ str((self.dateEdit.date()).toPyDate()) +"'")# lấy ngày ở dạng Qdate và chuyển sang obj datetime.date trong python = topydate
        result = cursor.fetchall()
        r=0
        c=0
        for row_number,row_data in enumerate(result):
            r+=1
            c=0
            for column_number,data in enumerate(row_data):
                c+=1
        self.REPORTTAB4.setColumnCount(c)

        for row_number,row_data in enumerate(result):
            self.REPORTTAB4.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTTAB4.setItem(row_number,column_number,QTableWidgetItem(str(data)))

        self.REPORTTAB4.setHorizontalHeaderLabels(['Id','Name','Date'])        
        self.REPORTTAB4.setColumnWidth(0,10)
        self.REPORTTAB4.setColumnWidth(1,60)
        self.REPORTTAB4.setColumnWidth(2,70)
        self.REPORTTAB4.verticalHeader().setVisible(False)
      
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()    