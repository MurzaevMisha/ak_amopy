import imaplib
import smtplib
import email.message
import os.path
import datetime
import xlrd
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.utils import make_grid
import time
import os
import face_recognition
from PIL import Image, ImageDraw
import pickle
from cv2 import cv2
import cv2
import torch
import torchvision
import torch.nn as nn
import sys
import mimetypes
from tqdm import tqdm
import numpy as np
from email import encoders
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase

YA_HOST = "smtp.gmail.com"
YA_PORT = 
YA_USER = ""
YA_PASSWORD = ""

pause_time = 5000


def get_cam():
    global count

        # подключились к почте и логинимся
    imap = imaplib.IMAP4_SSL(YA_HOST)
    imap.login(YA_USER, YA_PASSWORD)
    status, select_data = imap.select()
        # nmessages = select_data[0].decode('utf-8')

        # от кого письмо
    status, search_data = imap.search(None, 'FROM', 'akaemopy@gmail.com')

    for msg_id in reversed(search_data[0].split()):
        status, msg_data = imap.fetch(msg_id, '(RFC822)')
        # включает в себя заголовки и альтернативные полезные нагрузки
        mail = email.message_from_bytes(msg_data[0][1])

        if mail.is_multipart():
            filelist = []
            path = './data/none/none/'
            if not os.path.exists(path):
                os.makedirs(path)
            for part in mail.walk():
                content_type = part.get_content_type()
                filename = part.get_filename()
                if filename:
                    print(content_type)
                    print(filename)
                    if 'cam.jpg' == filename:
                        filelist.append(filename)
                        count = 1
                        emotionall = filename.rsplit( ".", 1 )[ 0 ]
                        print('Закачали файл: ', filename)

                        with open(path+part.get_filename(), 'wb') as new_file:
                            new_file.write(part.get_payload(decode=True))
            break
    imap.expunge()
    imap.logout()

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
        msg["Subject"] = "em"
        if text:
            msg.attach(MIMEText(text))

        if template:
            msg.attach(MIMEText(template, "html"))
        print("Collecting...")
        for file in tqdm(os.listdir("emo_face")):
            time.sleep(0.4)
            filename = os.path.basename(file)
            ftype, encoding = mimetypes.guess_type(file)
            file_type, subtype = ftype.split("/")
            with open(f"emo_face/{file}", "rb") as f:
                file = MIMEBase(file_type, subtype)
                file.set_payload(f.read())
                encoders.encode_base64(file)
            file.add_header('content-disposition', 'emo_face', filename=filename)
            msg.attach(file)
        print("Sending...")
        server.sendmail(to, to, msg.as_string())
        return "The message was sent successfully!"
    except Exception as _ex:
        return f"{_ex}\nCheck your login or password please!"
def main():
    data_dir = './data'
    classes_valid = os.listdir(data_dir + "/none")
    classes_train = ['Fear', 'Sad', 'Angry', 'Happy', 'Neutral']

    valid_tfms = tt.Compose([tt.Grayscale(num_output_channels=1), tt.Resize(48), tt.ToTensor()])
    valid_ds = ImageFolder(data_dir + "/none", valid_tfms)
    valid_dl = DataLoader(valid_ds, 200*2, num_workers=3, pin_memory=True)

    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(data, device):
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            return len(self.dl)

    device = torch.device('cpu')

    def accuracy(outputs, labels):
        global ura_result
        _, preds = torch.max(outputs, dim=1)
        values, ura_result = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                  nn.BatchNorm2d(out_channels), 
                  nn.ELU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    class ImageClassificationBase(nn.Module):
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)
            loss = F.cross_entropy(out, labels)
            acc = accuracy(out, labels)
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            return a

    class InceptionV3(ImageClassificationBase):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            
            self.conv1 = conv_block(in_channels, 128)
            self.conv2 = conv_block(128, 128, pool=True)
            self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
            self.drop1 = nn.Dropout(0.5)
            
            self.conv3 = conv_block(128, 256)
            self.conv4 = conv_block(256, 256, pool=True)
            self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
            self.drop2 = nn.Dropout(0.5)
            
            self.conv5 = conv_block(256, 512)
            self.conv6 = conv_block(512, 512, pool=True)
            self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
            self.drop3 = nn.Dropout(0.5)
            
            self.classifier = nn.Sequential(nn.MaxPool2d(6), 
                                            nn.Flatten(),
                                            nn.Linear(512, num_classes))
            
        def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.drop1(out)
            
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.drop2(out)
            
            out = self.conv5(out)
            out = self.conv6(out)
            out = self.res3(out) + out
            out = self.drop3(out)
            
            out = self.classifier(out)
            return out

    model = to_device(ResNet(1, len(classes_train)), device)
    model.load_state_dict(torch.load('./models/emotion_detection.pth'))


    def evaluate(model, val_loader):
        model.eval()
        outputs = ([model.validation_step(batch) for batch in val_loader])
        return model.validation_epoch_end(outputs)

    runing = evaluate(model, valid_dl)


labels_map = {0 : 'Angry',
            1 : 'Fear',
            2 : 'Happy',
            3 : 'Neutral',
            4 : 'Sad',
            }

stop_run = 0

def get_emotinal():
    global emott
    main()

    emott = labels_map[int(ura_result[0])]

    emo_file = open('./emo_face/'+ emott + ".txt", "w+")
    emo_file.close()

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

if __name__ == '__main__':
    while True:
        count = 0
        while not count == 1:
            get_cam()
        get_emotinal()
        text = ''
        template = ''
        print(send_email(text=text, template=template))
        os.remove('./emo_face/'+ emott + ".txt")
        del_email()
