/**
 * Error Banner component
 */

import React from 'react';
import './ErrorBanner.css';

function ErrorBanner({ message }) {
  return (
    <div className="error-banner">
      ⚠️ {message}
    </div>
  );
}

export default ErrorBanner;
