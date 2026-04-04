/**
 * Central error handler — catches any error thrown in route handlers.
 * Returns a clean JSON response instead of crashing the server.
 */
module.exports = (err, req, res, next) => {
  console.error(`[ERROR] ${req.method} ${req.path}:`, err.message);
  res.status(err.status || 500).json({
    success: false,
    error  : err.message || 'Internal server error'
  });
};
