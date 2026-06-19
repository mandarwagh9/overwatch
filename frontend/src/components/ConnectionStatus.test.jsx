/**
 * Smoke test for the ConnectionStatus presentational component.
 * Exercises the React Testing Library + jest-dom stack end-to-end.
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import ConnectionStatus from './ConnectionStatus';

describe('ConnectionStatus', () => {
  it('shows the connected state', () => {
    const { container } = render(<ConnectionStatus isConnected={true} />);
    expect(screen.getByText(/CONNECTED/i)).toBeInTheDocument();
    expect(container.firstChild).toHaveClass('connection-status', 'connected');
  });

  it('shows the disconnected state', () => {
    const { container } = render(<ConnectionStatus isConnected={false} />);
    expect(screen.getByText(/DISCONNECTED/i)).toBeInTheDocument();
    expect(container.firstChild).toHaveClass('connection-status', 'disconnected');
  });
});
