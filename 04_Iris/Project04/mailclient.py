from email.mime.base import MIMEBase
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders
import os
import json
import smtplib
import socks

MAINFOLDER = Path(__file__).parent
os.chdir(MAINFOLDER)

class Mailclient:
    def __init__(self):
        self.username = ""
        self.password = ""

        self.get_credentials()

    def get_credentials(self):
        filepath = MAINFOLDER / "config.json"

        with open(filepath,"r",encoding="UTF-8") as file:
            cfg = json.load(file)

        self.username = cfg["username"]
        self.pw = cfg["pw"]


    def send(self, subject, body, to = "", attachment_name = ""):
        if to == "":
            to = self.username
        
        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = to
        msg["Subject"] = subject

        #Body
        msg.attach(MIMEText(body, "plain"))

        if attachment_name != "":
            filepath = MAINFOLDER / attachment_name

            part = MIMEBase("application", "octet-stream")
            with open(filepath, "rb") as attachment:    
                part.set_payload(attachment.read()) #needs attachment to be open
            encoders.encode_base64(part)
            part.add_header(f"Content-Disposition", f"attachment; filename = {attachment_name}")
            msg.attach(part)

        self.transfer_message(msg)

    def transfer_message(self,msg):
        try:
            # s = socks.socksocket()
            # s.set_proxy(socks.HTTP, 'http://proxywbs', 3128)
            # s.connect(("www.google.com", 80))

            server = smtplib.SMTP("smtp.office365.com", 25) #587
            server.ehlo() #starts the connection
            server.starttls()
            server.login(self.username,self.pw)
            
            text = msg.as_string()

            server.sendmail(msg["From"], msg["To"], text)

            server.quit() #instaed with with?

        except Exception as e:
            print("Something went wrong")
            print(e)
        
        finally:
            pass
