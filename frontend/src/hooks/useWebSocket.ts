import { useCallback, useEffect, useRef, useState } from 'react';
export interface WebSocketMessage {
  type: 'analysis_result' | 'system_status' | 'error';
  data: any;
  timestamp: number;
}
interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}
export function useWebSocket({
  url,
  onMessage,
  onConnect,
  onDisconnect,
  onError,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5
}: UseWebSocketOptions) {
  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const connect = useCallback(() => {
    try {
      if (ws.current?.readyState === WebSocket.OPEN) {
        return;
      }
      setConnectionState('connecting');
      ws.current = new WebSocket(url);
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setConnectionState('connected');
        reconnectCount.current = 0;
        onConnect?.();
      };
      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setConnectionState('disconnected');
        onDisconnect?.();
        if (reconnectCount.current < maxReconnectAttempts) {
          reconnectCount.current++;
          console.log(`Attempting to reconnect (${reconnectCount.current}/${maxReconnectAttempts})...`);
          reconnectTimer.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else {
          console.log('Max reconnection attempts reached');
          setConnectionState('error');
        }
      };
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState('error');
        onError?.(error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionState('error');
    }
  }, [url, onMessage, onConnect, onDisconnect, onError, reconnectInterval, maxReconnectAttempts]);
  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    setConnectionState('disconnected');
  }, []);
  const sendMessage = useCallback((message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);
  const reconnect = useCallback(() => {
    disconnect();
    reconnectCount.current = 0;
    setTimeout(connect, 1000);
  }, [disconnect, connect]);
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);
  return {
    connectionState,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    reconnect,
    isConnected: connectionState === 'connected'
  };
}