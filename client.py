import smtplib
import imaplib
import os
import sys
import time
import mimetypes
from random import choice
import pygame
from mutagen.mp3 import MP3
from cv2 import cv2
import face_recognition
from pyfiglet import Figlet
from tqdm import tqdm
import email
from PIL import Image, ImageDraw
from email import encoders
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase

YA_HOST = ""
YA_PORT = 
YA_USER = ""
YA_PASSWORD = ""


def get_face():
    cap = cv2.VideoCapture(0)

    for i in range(30):
        cap.read()
      
    ret, frame = cap.read()

    cv2.imwrite('cam.jpg', frame)   


    def extracting_faces(img_path):
        count = 0
        size = 48, 48
        faces = face_recognition.load_image_file(img_path)
        faces_locations = face_recognition.face_locations(faces)

        for face_location in faces_locations:
            top, right, bottom, left = face_location

            face_img = faces[top:bottom, left:right]
            pil_img = Image.fromarray(face_img)
            pil_img.thumbnail(size)
            pil_img.save(f"./attachments/cam.jpg")
            count += 1

    extracting_faces('cam.jpg')


def send_email(text=None, template=None):
    sender = ""
    to = ''
    password = ""
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    try:
        with open(template) as file:
            template = file.read()
    except IOError:
        template = None
    try:
        server.login(sender, password)
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = "cam.jpg"
        if text:
            msg.attach(MIMEText(text))

        if template:
            msg.attach(MIMEText(template, "html"))
        print("Collecting...")
        for file in tqdm(os.listdir("attachments")):
            time.sleep(0.4)
            filename = os.path.basename(file)
            ftype, encoding = mimetypes.guess_type(file)
            file_type, subtype = ftype.split("/")
            with open(f"attachments/{file}", "rb") as f:
                file = MIMEBase(file_type, subtype)
                file.set_payload(f.read())
                encoders.encode_base64(file)
            file.add_header('content-disposition', 'attachment', filename=filename)
            msg.attach(file)
        print("Sending...")
        server.sendmail(to, to, msg.as_string())
        return "The message was sent successfully!"
    except Exception as _ex:
        return f"{_ex}\nCheck your login or password please!"
def main():
    font_text = Figlet(font="slant")
    print(font_text.renderText("SEND EMAIL"))
    text = ''
    template = ''
    print(send_email(text=text, template=template))

def get_emotinal():
    global count, emotionall

        # подключились к почте и логинимся
    imap = imaplib.IMAP4_SSL(YA_HOST)
    imap.login(YA_USER, YA_PASSWORD)
    status, select_data = imap.select()
        # nmessages = select_data[0].decode('utf-8')

        # от кого письмо
    status, search_data = imap.search(None, 'FROM', 'ak.emopy.server@gmail.com')

    for msg_id in reversed(search_data[0].split()):
        status, msg_data = imap.fetch(msg_id, '(RFC822)')
        # включает в себя заголовки и альтернативные полезные нагрузки
        mail = email.message_from_bytes(msg_data[0][1])

        if mail.is_multipart():
            filelist = []
            path = './get_emo'
            if not os.path.exists(path):
                os.makedirs(path)
            for part in mail.walk():
                content_type = part.get_content_type()
                filename = part.get_filename()
                if filename:
                    print(content_type)
                    print(filename)
                    if '.txt' in filename:
                        filelist.append(filename)
                        count = 1
                        emotionall = filename.rsplit( ".", 1 )[ 0 ]
                        print(emotionall)
                        print('Закачали файл: ', filename)
                        with open(path+part.get_filename(), 'wb') as new_file:
                            new_file.write(part.get_payload(decode=True))
                        

            break
    imap.expunge()
    imap.logout()

def get_way(emotinal):
    directory = 'music/' + emotinal
    files = os.listdir(directory)
    trek = choice(files)
    directory += '/' + trek
    print(emotinal, ' ', trek)
    return directory

def play_music(emotinal):
    global time_song
    music = get_way(emotinal)
    time_song = MP3(music)
    pygame.init()
    song = pygame.mixer.Sound(music)
    clock = pygame.time.Clock()
    song.play()
    time.sleep(time_song.info.length)
    pygame.quit()

def del_email():
    server = ""
    port = ""
    login = ""
    password = ""
    putdir="/home/pavel/"

    print ("- подключаемся к ",server)
    mail = imaplib.IMAP4_SSL(server)
    print ("-- логинимся")
    mail.login(login, password)
    mail.list()
    print ("-- подключаемся к inbox")
    mail.select("inbox")
    print ("-- получаем UID последнего письма");
    result, data = mail.uid('search', None, "ALL")       
    try:
        latest_email_uid = data[0].split()[-1]     
    except IndexError:
        print("-- писем нет!")
        exit(0)
    result, data = mail.uid('fetch', latest_email_uid, '(RFC822)')
    raw_email = data[0][1]
    try:
        email_message = email.message_from_string(raw_email)   
    except TypeError:
        email_message = email.message_from_bytes(raw_email)
    print ("--- нашли письмо от: ",email.header.make_header(email.header.decode_header(email_message['From'])))
    for part in email_message.walk():
        print(part.get_content_type())
        if "application" in part.get_content_type() :       
            filename = part.get_filename()
            filename=str(email.header.make_header(email.header.decode_header(filename)))
            if not(filename): filename = "test.txt"          
            print ("---- нашли вложение ",filename)        
            fp = open(os.path.join(putdir, filename), 'wb')
            fp.write(part.get_payload(decode=1))
            fp.close
    print ("-- удаляем письмо");
    mail.uid('STORE', latest_email_uid , '+FLAGS', r'(\Deleted)')  
    mail.expunge()
    return 1

def run():
    global count, emotinall
    get_face()
    main()
    count = 0
    while not count == 1: 
        get_emotinal()
    del_email()
    play_music(emotionall)

if __name__ == "__main__":
    while True:
        get_face()
        main()
        count = 0
        while not count == 1: 
            get_emotinal()
        del_email()
        play_music(emotionall)
