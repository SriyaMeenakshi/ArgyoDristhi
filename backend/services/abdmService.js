const axios = require('axios');

/**
 * ABDM (Ayushman Bharat Digital Mission) integration.
 * Guide Section 7.4.
 *
 * Setup:
 *   1. Register at sandbox.abdm.gov.in (free, for development)
 *   2. Read API docs at ndhm.gov.in/apis
 *   3. Set ABDM_CLIENT_ID + ABDM_CLIENT_SECRET in .env
 *
 * Flow:
 *   1. Get access token from ABDM auth endpoint
 *   2. Convert screening to FHIR R4 DiagnosticReport bundle
 *   3. POST bundle to ABDM Health Information Exchange (HIE)
 */

let cachedToken = null;
let tokenExpiry = 0;

/**
 * Get a valid ABDM access token (cached until expiry).
 */
async function getAccessToken() {
  if (cachedToken && Date.now() < tokenExpiry) return cachedToken;

  const resp = await axios.post(
    `${process.env.ABDM_BASE_URL}/sessions`,
    {
      clientId    : process.env.ABDM_CLIENT_ID,
      clientSecret: process.env.ABDM_CLIENT_SECRET
    }
  );

  cachedToken = resp.data.accessToken;
  // Tokens typically expire in 30 min; refresh 2 min early
  tokenExpiry = Date.now() + (resp.data.expiresIn - 120) * 1000;
  return cachedToken;
}

/**
 * Build a FHIR R4 Bundle (DiagnosticReport) from a screening result.
 * Guide: "Your backend converts a completed screening into a FHIR-compliant
 *         JSON document and sends it to the ABDM HIE."
 */
function buildFhirBundle(patient, screening) {
  const now = new Date().toISOString();
  const bundleId = `arogya-${screening.screeningId}`;

  // Map risk scores to LOINC-like observation codes
  const observations = [
    { code: '59408-5', display: 'Hematological risk',   value: screening.hematologicalRisk },
    { code: '15074-8', display: 'Metabolic risk',        value: screening.metabolicRisk },
    { code: '33914-3', display: 'Renal risk',            value: screening.renalRisk },
    { code: '1742-6',  display: 'Hepatic risk',          value: screening.hepaticRisk },
    { code: '55284-4', display: 'Cardiovascular risk',   value: screening.cardiovascularRisk },
    { code: '8661-1',  display: 'Dermatological risk',   value: screening.dermatologicalRisk },
    { code: '75303-8', display: 'Nutritional risk',      value: screening.nutritionalRisk }
  ].map((obs, i) => ({
    fullUrl : `urn:uuid:obs-${i}`,
    resource: {
      resourceType: 'Observation',
      id          : `obs-${i}`,
      status      : 'final',
      code        : {
        coding: [{ system: 'http://loinc.org', code: obs.code, display: obs.display }]
      },
      subject     : { reference: `Patient/${patient.abhaId}` },
      effectiveDateTime: screening.date,
      valueQuantity: {
        value : parseFloat((obs.value * 100).toFixed(1)),
        unit  : '%',
        system: 'http://unitsofmeasure.org',
        code  : '%'
      },
      interpretation: [{
        coding: [{
          system : 'http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation',
          code   : obs.value >= 0.70 ? 'H' : obs.value >= 0.35 ? 'N' : 'L',
          display: obs.value >= 0.70 ? 'High' : obs.value >= 0.35 ? 'Normal' : 'Low'
        }]
      }]
    }
  }));

  return {
    resourceType: 'Bundle',
    id          : bundleId,
    type        : 'document',
    timestamp   : now,
    entry       : [
      // Patient resource
      {
        fullUrl : `Patient/${patient.abhaId}`,
        resource: {
          resourceType: 'Patient',
          id          : patient.abhaId,
          identifier  : [{
            system: 'https://healthid.ndhm.gov.in',
            value : patient.abhaId
          }],
          name: [{ text: patient.name }],
          gender: patient.sex?.toLowerCase() === 'female' ? 'female' : 'male',
          birthDate: estimateBirthYear(patient.age)
        }
      },
      // DiagnosticReport resource
      {
        fullUrl : `urn:uuid:report-${screening.screeningId}`,
        resource: {
          resourceType: 'DiagnosticReport',
          id          : `report-${screening.screeningId}`,
          status      : 'final',
          code        : {
            coding: [{
              system : 'http://loinc.org',
              code   : '81247-9',
              display: 'Master HL7 genetic variant reporting panel'
            }],
            text: 'ArogyaDrishti Multi-Modal Health Screening'
          },
          subject   : { reference: `Patient/${patient.abhaId}` },
          effectiveDateTime: screening.date,
          issued    : now,
          result    : observations.map((_, i) => ({ reference: `urn:uuid:obs-${i}` })),
          conclusion: `Overall risk: ${screening.overallRiskLevel.toUpperCase()}. Screened by ASHA Worker ${patient.ashaWorkerId || 'Unknown'}.`
        }
      },
      // All observations
      ...observations
    ]
  };
}

/**
 * POST the FHIR bundle to ABDM HIE.
 */
async function syncToAbdm(fhirBundle, abhaId) {
  const token = await getAccessToken();

  const resp = await axios.post(
    `${process.env.ABDM_BASE_URL}/health-information/hip/request`,
    fhirBundle,
    {
      headers: {
        'Authorization'  : `Bearer ${token}`,
        'Content-Type'   : 'application/fhir+json',
        'X-CM-ID'        : 'sbx',    // sandbox CM ID
        'ABDM-Patient-Id': abhaId
      }
    }
  );

  return resp.data;
}

function estimateBirthYear(age) {
  const year = new Date().getFullYear() - age;
  return `${year}-01-01`;
}

module.exports = { buildFhirBundle, syncToAbdm };
