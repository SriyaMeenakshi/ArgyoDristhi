/**
 * Seed PHC doctor contacts into Firestore.
 * Run once after Firebase is set up:
 *   node seedDoctors.js
 *
 * Add real doctor numbers from your district NHM directory.
 * Each document links a village to the nearest PHC doctor's WhatsApp.
 */
require('dotenv').config();
const { collections } = require('./config/firebase');

const doctors = [
  {
    name           : 'Dr. Ravi Kumar',
    phcName        : 'PHC Pedanandipadu',
    village        : 'Pedanandipadu',
    district       : 'Krishna',
    state          : 'Andhra Pradesh',
    whatsappNumber : '+919876543210',
    phone          : '+919876543210'
  },
  {
    name           : 'Dr. Sunitha Rao',
    phcName        : 'PHC Vuyyuru',
    village        : 'Vuyyuru',
    district       : 'Krishna',
    state          : 'Andhra Pradesh',
    whatsappNumber : '+919876543211',
    phone          : '+919876543211'
  },
  {
    name           : 'Dr. Venkata Reddy',
    phcName        : 'PHC Nandigama',
    village        : 'Nandigama',
    district       : 'NTR',
    state          : 'Andhra Pradesh',
    whatsappNumber : '+919876543212',
    phone          : '+919876543212'
  }
  // Add more from: https://hmis.nhp.gov.in (NHM facility directory)
];

async function seed() {
  console.log(`Seeding ${doctors.length} PHC doctor record(s)...`);
  for (const doctor of doctors) {
    const docRef = collections.doctors.doc();
    await docRef.set({ ...doctor, createdAt: new Date().toISOString() });
    console.log(`  Added: ${doctor.name} — ${doctor.phcName}`);
  }
  console.log('Done. Run once only.');
  process.exit(0);
}

seed().catch(err => {
  console.error('Seed failed:', err.message);
  process.exit(1);
});
