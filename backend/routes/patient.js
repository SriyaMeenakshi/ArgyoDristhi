const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { collections } = require('../config/firebase');

const router = express.Router();

/**
 * POST /api/patient/register
 * Called by Android app on Screen 2 (Patient Registration).
 * Creates a new patient document in Firestore.
 *
 * Body: { name, age, sex, abhaId, village, isPregnant, ashaWorkerId }
 */
router.post('/register', async (req, res, next) => {
  try {
    const { name, age, sex, abhaId, village, isPregnant, ashaWorkerId } = req.body;

    if (!name || !age || !sex) {
      return res.status(400).json({ success: false, error: 'name, age and sex are required' });
    }

    // Check if patient with this ABHA ID already exists
    if (abhaId) {
      const existing = await collections.patients
        .where('abhaId', '==', abhaId)
        .limit(1)
        .get();
      if (!existing.empty) {
        const doc = existing.docs[0];
        return res.json({ success: true, patientId: doc.id, existing: true });
      }
    }

    const patientId = uuidv4();
    const patient = {
      patientId,
      name,
      age        : Number(age),
      sex,
      abhaId     : abhaId || '',
      village    : village || '',
      isPregnant : Boolean(isPregnant),
      ashaWorkerId: ashaWorkerId || '',
      createdAt  : new Date().toISOString()
    };

    await collections.patients.doc(patientId).set(patient);
    res.status(201).json({ success: true, patientId });

  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/patient/:abha_id
 * Retrieves a patient's full profile and screening history by ABHA ID.
 * Used by PHC doctor to pull up longitudinal records.
 */
router.get('/:abha_id', async (req, res, next) => {
  try {
    const { abha_id } = req.params;

    const snapshot = await collections.patients
      .where('abhaId', '==', abha_id)
      .limit(1)
      .get();

    if (snapshot.empty) {
      return res.status(404).json({ success: false, error: 'Patient not found' });
    }

    const patient = snapshot.docs[0].data();
    const patientId = patient.patientId;

    // Fetch screening history for this patient
    const screeningsSnap = await collections.screenings
      .where('patientId', '==', patientId)
      .orderBy('date', 'desc')
      .get();

    const screenings = screeningsSnap.docs.map(d => d.data());

    res.json({ success: true, patient, screenings });

  } catch (err) {
    next(err);
  }
});

module.exports = router;
