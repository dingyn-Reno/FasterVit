import smtplib
from email.mime.text import MIMEText

def send(text):
    # 设置服务器所需信息
    # 邮箱服务器地址
    mail_host = 'smtp.exmail.qq.com'
    # 163用户名
    mail_user = 'dingyn@djinni.club'
    # 密码(部分邮箱为授权码)
    mail_pass = 'Ding1234'
    # 邮件发送方邮箱地址
    sender = 'dingyn@djinni.club'
    # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
    receivers = ['dingyn@djinni.club']

    # 设置email信息
    # 邮件内容设置
    message = MIMEText(text, 'plain', 'utf-8')
    # 邮件主题
    message['Subject'] = 'title'
    # 发送方信息
    message['From'] = sender
    # 接受方信息
    message['To'] = receivers[0]

    # 登录并发送邮件
    try:
        smtpObj = smtplib.SMTP()
        # 连接到服务器
        smtpObj.connect(mail_host, 25)
        # 登录到服务器
        smtpObj.login(mail_user, mail_pass)
        # 发送
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        # 退出
        smtpObj.quit()
        print('success')
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误
