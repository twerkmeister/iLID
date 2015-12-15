import smtplib

EMAIL_HOST = "mail3.hpi.uni-potsdam.de"
EMAIL_PORT = 25
FROM_EMAIL = "MLboy@hpi.de"
TO_EMAIL = ["tom.herold@student.hpi.de", "thomas.werkmeister@student.hpi.de"]

def send_email_notification(precision):

    subject = "Training has finished"
    message_body = "Hello DeepGuys, \n\n The training has finined with a precision@1 {0}. \n\n Call me maybe, MLBoy".format(precision)

    msg = "From: {from_name} <{from_addr}>\nTo: DeepGuys <{to_addr}>\nSubject: {subject_mail}\n{message}".format(
                    from_name=FROM_EMAIL.split('@')[0],
                    from_addr=FROM_EMAIL,
                    to_addr=", ".join(TO_EMAIL),
                    subject_mail=subject,
                    message=message_body)



    server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
    server.ehlo()
    server.sendmail(FROM_EMAIL, TO_EMAIL, msg)
    server.quit()
