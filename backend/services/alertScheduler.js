const cron = require('node-cron');
const { collections } = require('../config/firebase');
const twilioService = require('./twilioService');

/**
 * 48-hour follow-up cron job.
 * Guide Section 7.3:
 *   "If the doctor has not acknowledged within 48 hours
 *    (phc_acknowledged stays false), your backend sends a follow-up alert."
 *
 * Runs every hour and checks for unacknowledged alerts older than 48 hours.
 */

cron.schedule('0 * * * *', async () => {
  console.log('[Scheduler] Checking for unacknowledged alerts...');
  try {
    const cutoff = new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString();

    const snapshot = await collections.alerts
      .where('phcAcknowledged', '==', false)
      .where('sentAt', '<', cutoff)
      .get();

    if (snapshot.empty) {
      console.log('[Scheduler] No unacknowledged alerts past 48 hours.');
      return;
    }

    console.log(`[Scheduler] Found ${snapshot.size} unacknowledged alert(s) — sending follow-ups.`);

    for (const doc of snapshot.docs) {
      const alert = doc.data();
      try {
        const followUpBody = [
          `FOLLOW-UP — ArogyaDrishti Alert (48h unacknowledged)`,
          `Patient alert sent on ${new Date(alert.sentAt).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })} has not been acknowledged.`,
          `Please review and acknowledge to stop further reminders.`,
          `Original message:`,
          alert.whatsappBody
        ].join('\n');

        await twilioService.sendWhatsApp(alert.whatsappTo, followUpBody);

        // Mark that a follow-up was sent (add followUpSentAt field)
        await collections.alerts.doc(doc.id).update({
          followUpSentAt: new Date().toISOString()
        });

        console.log(`[Scheduler] Follow-up sent for alert ${alert.alertId}`);
      } catch (err) {
        console.error(`[Scheduler] Failed to send follow-up for ${alert.alertId}:`, err.message);
      }
    }
  } catch (err) {
    console.error('[Scheduler] Error checking alerts:', err.message);
  }
});

console.log('[Scheduler] 48-hour follow-up cron job started (runs every hour).');
