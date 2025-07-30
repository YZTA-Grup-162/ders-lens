import React from 'react';
interface ConsentModalProps {
  isOpen: boolean;
  onAccept: () => void;
  onDecline: () => void;
  onClose: () => void;
}
export const ConsentModal: React.FC<ConsentModalProps> = ({
  isOpen,
  onAccept,
  onDecline,
  onClose
}) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-900">
            Camera Access Consent
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="mb-6">
          <p className="text-gray-700 mb-4">
            DersLens requires access to your camera to analyze your attention and engagement levels 
            during the learning session.
          </p>
          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <h3 className="font-semibold text-blue-900 mb-2">
              Privacy Guarantee
            </h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Your video data is processed locally on your device</li>
              <li>• No video footage is stored or transmitted</li>
              <li>• Only anonymized attention scores are collected</li>
              <li>• You can withdraw consent at any time</li>
            </ul>
          </div>
          <div className="bg-yellow-50 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-900 mb-2">
              Data Processing
            </h3>
            <p className="text-sm text-yellow-800">
              By continuing, you consent to the processing of your biometric data for educational 
              analytics purposes in accordance with our Privacy Policy and GDPR requirements.
            </p>
          </div>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={onDecline}
            className="flex-1 px-4 py-2 text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300 transition-colors"
          >
            Decline
          </button>
          <button
            onClick={onAccept}
            className="flex-1 px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
          >
            Accept & Continue
          </button>
        </div>
        <div className="mt-4 text-center">
          <a
            href="/privacy-policy"
            className="text-sm text-blue-600 hover:text-blue-700 underline"
          >
            View Privacy Policy
          </a>
        </div>
      </div>
    </div>
  );
};