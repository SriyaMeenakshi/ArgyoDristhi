const twilio = require('twilio');

/**
 * Twilio service — WhatsApp + SMS alerts.
 * Guide Section 7.3.
 *
 * WhatsApp sandbox setup (one-time):
 *   1. Go to twilio.com → Messaging → WhatsApp → Sandbox
 *   2. The sandbox number is +14155238886
 *   3. Every recipient must first send "join <sandbox-keyword>" to opt in
 *
 * Free trial gives $15.50 credit (~800 SMS or ~500 WhatsApp messages).
 */

const client = twilio(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN
);

/**
 * Send a WhatsApp message to the PHC doctor.
 * @param {string} to   - e.g. "whatsapp:+919876543210"
 * @param {string} body - message text (guide template — 6 lines)
 * @returns {string}    - Twilio message SID
 */
async function sendWhatsApp(to, body) {
  const message = await client.messages.create({
    from: process.env.TWILIO_WHATSAPP_FROM,
    to,
    body
  });
  console.log(`WhatsApp sent to ${to} — SID: ${message.sid}`);
  return message.sid;
}

/**
 * Send an SMS to the patient.
 * @param {string} to   - patient phone number e.g. "+919876543210"
 * @param {string} body - message in regional language (Telugu default for AP)
 * @returns {string}    - Twilio message SID
 */
async function sendSms(to, body) {
  const message = await client.messages.create({
    from: process.env.TWILIO_SMS_FROM,
    to,
    body
  });
  console.log(`SMS sent to ${to} — SID: ${message.sid}`);
  return message.sid;
}

/**
 * Regional language SMS templates — guide Section 7.3.
 * Used for the follow-up reminder as well as the initial alert.
 */
const smsTemplates = {
  telugu: (patientName, doctorName) =>
    `ArogyaDrishti: ${patientName} గారికి, మీ ఆరోగ్య పరీక్షలో ముఖ్యమైన సమస్య గుర్తించబడింది. ` +
    `దయచేసి వెంటనే సమీప PHC ని సందర్శించి ${doctorName} ని కలవండి.`,

  hindi: (patientName, doctorName) =>
    `ArogyaDrishti: ${patientName} जी, आपकी स्वास्थ्य जांच में एक महत्वपूर्ण समस्या पाई गई है। ` +
    `कृपया तुरंत नजदीकी PHC जाएं और ${doctorName} से मिलें।`,

  tamil: (patientName, doctorName) =>
    `ArogyaDrishti: ${patientName} அவர்களே, உங்கள் உடல் பரிசோதனையில் முக்கியமான பிரச்சினை கண்டறியப்பட்டது. ` +
    `உடனே அருகிலுள்ள PHC க்கு சென்று ${doctorName} ஐ சந்தியுங்கள்.`,

  english: (patientName, doctorName) =>
    `ArogyaDrishti: Dear ${patientName}, a health risk was detected in your screening. ` +
    `Please visit your nearest PHC immediately and ask for ${doctorName}. Show this message.`
};

module.exports = { sendWhatsApp, sendSms, smsTemplates };
