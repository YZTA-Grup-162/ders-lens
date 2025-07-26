import { useEffect, useState } from 'react';
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: any;
}
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetError: () => void }>;
}
export function ErrorBoundary({ children, fallback: Fallback }: ErrorBoundaryProps) {
  const [state, setState] = useState<ErrorBoundaryState>({
    hasError: false,
    error: null,
    errorInfo: null
  });
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      console.error('Global error caught:', event.error);
      setState({
        hasError: true,
        error: event.error,
        errorInfo: { componentStack: event.filename + ':' + event.lineno }
      });
    };
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason);
      setState({
        hasError: true,
        error: new Error(event.reason),
        errorInfo: { componentStack: 'Promise rejection' }
      });
    };
    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);
  const resetError = () => {
    setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };
  if (state.hasError && state.error) {
    if (Fallback) {
      return <Fallback error={state.error} resetError={resetError} />;
    }
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center p-6">
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-8 max-w-lg w-full">
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 19c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h2 className="text-xl font-bold text-red-400 mb-2">Bir Hata Oluştu</h2>
            <p className="text-gray-300 text-sm mb-4">
              Uygulama beklenmedik bir hata ile karşılaştı.
            </p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-medium text-gray-300 mb-2">Hata Detayları:</h3>
            <pre className="text-xs text-red-300 overflow-auto">
              {state.error.message}
            </pre>
          </div>
          <div className="flex gap-3">
            <button
              onClick={resetError}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Tekrar Dene
            </button>
            <button
              onClick={() => window.location.reload()}
              className="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Sayfayı Yenile
            </button>
          </div>
        </div>
      </div>
    );
  }
  return <>{children}</>;
}