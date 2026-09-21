import { useState, useEffect } from 'react';
import { checkBackendHealth } from '../services/api';

export function useBackendStatus(pollIntervalMs = 30000) {
  const [status, setStatus] = useState('checking');
  const [data, setData] = useState(null);

  useEffect(() => {
    let isMounted = true;

    async function check() {
      const res = await checkBackendHealth();
      if (isMounted) {
        setStatus(res.status === 'ok' ? 'online' : 'offline');
        setData(res);
      }
    }

    check();
    const interval = setInterval(check, pollIntervalMs);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [pollIntervalMs]);

  return { status, data, isOnline: status === 'online' };
}
