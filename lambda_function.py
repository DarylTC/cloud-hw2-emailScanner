import json
import boto3
import os
import urllib.parse
import email
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

s3_client = boto3.client('s3')
ENDPOINT_NAME = 'sms-spam-classifier-mxnet-2022-11-26-18-28-30-807'

def send_email(from_email,email_receive_date,email_subject,email_body_short,classification,class_conf_score):
    SENDER = 'dc4676@nyu.edu' # must be verified in AWS SES Email
    RECIPIENT = from_email # must be verified in AWS SES Email
    AWS_REGION = 'us-east-1'
    SUBJECT = 'Your spam or ham email result at %s'%(email_receive_date)
    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = ("Hey")                
    # The HTML body of the email.
    htmlPTag = []

    body =''
    for p in htmlPTag:
        body += p

    BODY_HTML = '''
    <html>
    <head></head>
    <body>
    <h3>We received your email sent at %s with the subject %s.</h3>
    <br>
    <h3>Here is a 240 character sample of the email body:</h3>
    %s
    <br>
    <h3>The email was categorized as %s with a %f confidence.</h3>
    <br>
    <p> Thank you </p>
    </body>
    </html>
    '''%(email_receive_date,email_subject,email_body_short,classification,class_conf_score)    
    # The character encoding for the email.
    CHARSET = 'UTF-8'

    # Create a new SES resource and specify a region.
    ses = boto3.client('ses',region_name=AWS_REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = ses.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            ReplyToAddresses=['dc4676@nyu.edu'],
            Message={
                'Body': {
                    'Html': {
        
                        'Data': BODY_HTML
                    },
                    'Text': {
        
                        'Data': BODY_TEXT
                    },
                },
                'Subject': {

                    'Data': SUBJECT
                },
            },
            Source=SENDER
        )
        print('email sent!')
    except:
        raise

def lambda_handler(event, context):

    print(event)
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    # key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    email_receive_date = event['Records'][0]['eventTime']
    print('bucket: ' + bucket)
    print('email object name in S3: ' + key)
    print('email receive timestamp: ' + email_receive_date)
    


    # 1.extract email body (need to strip out new line char, \n)
        # need actual email body to test (using SES default one for now)
    getobj=s3_client.get_object(Bucket=bucket,Key=key)
    body=getobj['Body'].read().decode("utf-8")
    email_contents = email.message_from_string(body)
        
    # parsing email content 
    from_email = ''
    to_email = ''
    email_receive_date = ''
    email_subject = ''
    email_body = '' # limit to 240 characters

    from_email = email_contents.get('From')
    from_email = from_email[from_email.find('<') + 1:-1]
    to_email = email_contents.get('To')
    email_receive_date = email_contents.get('Date')
    email_subject = email_contents.get('Subject')
    print('from_email:' + from_email)
    print('to_email:' + to_email)
    print('email_receive_date:' + email_receive_date)
    print('email_subject:' + email_subject)

    if email_contents.is_multipart():
        for payload in email_contents.get_payload():
            if payload.get_content_type() == 'text/plain':
                email_body = payload.get_payload()
    else:
        email_body = email_contents.get_payload()
    email_body = email_body.replace("\r", " ").replace("\n", " ")

    print('email_body:' + email_body)
    
    
    # 2.call prediction endpoint to predict the email 
        # classification
        # classification_confidence_score (it is a %)
    # ENDPOINT_NAME = os.environ['PRED_URL']
    vocabulary_length=9013
    emailBody_for_processing=[email_body]
    runtime= boto3.client('runtime.sagemaker') 
    one_hot_test_messages = one_hot_encode(emailBody_for_processing, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    payload = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=payload)
    response_body = response['Body'].read().decode('utf-8')
    result = json.loads(response_body)
    classification = ''
    if(result["predicted_label"][0][0]==1):
            classification = ''+'SPAM'
            classification_confidence_score = result['predicted_probability'][0][0]*100
    else:
            classification = ''+'HAM'
            classification_confidence_score = result['predicted_probability'][0][0]*100
            
    print('classification: ' + classification)
    print('classification_confidence_score: ', classification_confidence_score)
    
    
    # 3.send reply to sender email with fixed body
        # Params needed:
            # email_receive_date
            # email_subject
            # email_body (240 chars)
    if len(email_body) > 240:
        email_body_short = email_body[:240]
    else:
        email_body_short = email_body

    send_email(from_email,email_receive_date,email_subject,email_body_short,classification,classification_confidence_score)
    

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
