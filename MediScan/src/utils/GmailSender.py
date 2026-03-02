import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict


def send_health_report_email(
    receiver_email: str,
    symptoms: List[str],
    disease: str,
    top3_predictions: List[Dict[str, str]],  # [{"disease": "...", "probability": "..."}]
    description: str,
    precautions: List[str],
    home_remedy: str,
    severity_level: str,
    reaction_advice: str,
    hospital_list: List[Dict[str, str]],  # [{"name": "...", "url": "..."}]
):
    """
    Sends a professional health report email.
    """

    sender_email = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    if not sender_email or not password:
        raise ValueError("Email credentials not found in environment variables.")

    prediction_section = ""
    if top3_predictions:
        prediction_section += """
        <h3 style="color:#884EA0;">Top 3 Possible Conditions (AI Confidence)</h3>
        <table width="100%" style="border-collapse: collapse;">
            <tr style="background-color:#F2F3F4;">
                <th align="left" style="padding:8px; border:1px solid #ddd;">Condition</th>
                <th align="left" style="padding:8px; border:1px solid #ddd;">Probability</th>
            </tr>
        """

        for pred in top3_predictions:
            prediction_section += f"""
            <tr>
                <td style="padding:8px; border:1px solid #ddd;">
                    {pred['disease']}
                </td>
                <td style="padding:8px; border:1px solid #ddd;">
                    {pred['probability']}
                </td>
            </tr>
            """

        prediction_section += "</table>"

    # -------------------------------
    # Hospital Section
    # -------------------------------
    hospital_section = ""
    if hospital_list:
        hospital_section += """
        <h3 style="color:#B03A2E;">Nearby Hospitals</h3>
        <ul>
        """

        for hospital in hospital_list:
            hospital_section += f"""
            <li>
                <b>{hospital['name']}</b><br>
                <a href="{hospital['url']}" style="color:#2E86C1; text-decoration:none;">
                    Open in Google Maps
                </a>
            </li>
            """

        hospital_section += "</ul>"

    body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color:#333;">
        <div style="max-width:600px; margin:auto; padding:20px;
                    border:1px solid #ddd; border-radius:10px;">

          <h2 style="color:#2E86C1; text-align:center;">
              🩺 MediScan Health Report
          </h2>

          <p>Hello,</p>
          <p>
              Based on the symptoms you provided, our AI system has generated
              the following health insights.
          </p>

          <h3 style="color:#117A65;">Submitted Symptoms</h3>
          <ul>
              {''.join(f"<li>{s}</li>" for s in symptoms)}
          </ul>

          <h3 style="color:#CB4335;">Possible Condition</h3>
          <p><b>{disease}</b></p>

          <h3 style="color:#6C3483;">Condition Overview</h3>
          <p>{description}</p>

          <h3 style="color:#AF601A;">Severity Level</h3>
          <p><b>{severity_level}</b></p>

          <h3 style="color:#1F618D;">Recommended Action</h3>
          <p>{reaction_advice}</p>

          <h3 style="color:#AF601A;">Precautions</h3>
          <ul>
              {''.join(f"<li>{p}</li>" for p in precautions)}
          </ul>

          <h3 style="color:#1F618D;">Suggested Home Remedy</h3>
          <p>{home_remedy}</p>

          {hospital_section}

          <hr style="margin-top:30px;">

          <p style="font-size:12px; color:#777;">
              ⚠️ Disclaimer: This report is AI-generated and is not a substitute
              for professional medical advice. Please consult a qualified doctor
              for accurate diagnosis and treatment.
          </p>

        </div>
      </body>
    </html>
    """


    message = MIMEMultipart("alternative")
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "🩺 MediScan – Your Personalized Health Report"

    message.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())

        print(f"✅ Email sent successfully to {receiver_email}")

    except Exception as e:
        print("❌ Failed to send email:", e)