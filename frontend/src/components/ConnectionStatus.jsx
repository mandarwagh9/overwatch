/**
 * Connection Status component
 */

import React from 'react';
import './ConnectionStatus.css';

function ConnectionStatus({ isConnected }) {
  return (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      {isConnected ? '🟢 CONNECTED' : '🔴 DISCONNECTED'}
    </div>
  );
}

export default ConnectionStatus;
