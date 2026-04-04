const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { collections } = require('../config/firebase');

const router = express.Router();

/**
 * POST /api/screening/save
 * Saves a completed screening result to Firestore.
 * Called by Android WorkManager when internet is available.
 *
 * Body: {
 *   patientId, symptoms,
 *   hematologicalRisk, metabolicRisk, renalRisk, hepaticRisk,
 *   cardiovascularRisk, dermatologicalRisk, nutritionalRisk,
 *   overallRiskLevel, ashaWorkerId, village
 * }
 */
router.post('/save', async (req, res, next) => {
  try {
    const {
      patientId,
      symptoms          = '',
      hematologicalRisk  = 0,
      metabolicRisk      = 0,
      renalRisk          = 0,
      hepaticRisk        = 0,
      cardiovascularRisk = 0,
      dermatologicalRisk = 0,
      nutritionalRisk    = 0,
      overallRiskLevel   = 'green',
      ashaWorkerId       = '',
      village            = ''
    } = req.body;

    if (!patientId) {
      return res.status(400).json({ success: false, error: 'patientId is required' });
    }

    const screeningId = uuidv4();
    const screening = {
      screeningId,
      patientId,
      date              : new Date().toISOString(),
      symptoms,
      hematologicalRisk : Number(hematologicalRisk),
      metabolicRisk     : Number(metabolicRisk),
      renalRisk         : Number(renalRisk),
      hepaticRisk       : Number(hepaticRisk),
      cardiovascularRisk: Number(cardiovascularRisk),
      dermatologicalRisk: Number(dermatologicalRisk),
      nutritionalRisk   : Number(nutritionalRisk),
      overallRiskLevel,
      ashaWorkerId,
      village,
      syncedToAbdm      : false
    };

    await collections.screenings.doc(screeningId).set(screening);
    res.status(201).json({ success: true, screeningId });

  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/screening/:id
 * Returns full details of a single screening.
 * Used by PHC doctor dashboard to review findings.
 */
router.get('/:id', async (req, res, next) => {
  try {
    const doc = await collections.screenings.doc(req.params.id).get();

    if (!doc.exists) {
      return res.status(404).json({ success: false, error: 'Screening not found' });
    }

    // Enrich with patient details
    const screening = doc.data();
    const patientDoc = await collections.patients.doc(screening.patientId).get();
    const patient = patientDoc.exists ? patientDoc.data() : null;

    res.json({ success: true, screening, patient });

  } catch (err) {
    next(err);
  }
});

module.exports = router;
