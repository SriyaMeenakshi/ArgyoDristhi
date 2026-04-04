require('dotenv').config();

const express = require('express');
const cors    = require('cors');

const patientRoutes   = require('./routes/patient');
const screeningRoutes = require('./routes/screening');
const alertRoutes     = require('./routes/alert');
const abdmRoutes      = require('./routes/abdm');
const errorHandler    = require('./middleware/errorHandler');

// Start cron jobs (48-hour follow-up alerts)
require('./services/alertScheduler');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// ── Routes ────────────────────────────────────────────────────────────────────
app.use('/api/patient',   patientRoutes);
app.use('/api/screening', screeningRoutes);
app.use('/api/alert',     alertRoutes);
app.use('/api/abdm',      abdmRoutes);

// GET /api/health — deployment health check
app.get('/api/health', (req, res) => {
  res.json({
    status : 'ok',
    service: 'ArogyaDrishti Backend',
    time   : new Date().toISOString()
  });
});

// ── Error handler (must be last) ─────────────────────────────────────────────
app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`ArogyaDrishti backend running on port ${PORT}`);
});
