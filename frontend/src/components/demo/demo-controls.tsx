import clsx from 'clsx';
import { AnimatePresence, motion } from 'framer-motion';
import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useDemoControls, useDemoHotkeys, useDemoStore } from '../../stores/demo-store';
import { NeonButton } from '../ui/neon-button';

interface DemoControlsProps {
  className?: string;
  visible?: boolean;
}

/**
 * DemoControls - Hidden hotkey-activated demo management interface
 */
export const DemoControls: React.FC<DemoControlsProps> = ({
  className,
  visible = false
}) => {
  const { t } = useTranslation();
  const {
    isPlaying,
    currentTime,
    playbackSpeed,
    play,
    pause,
    reset,
    setCurrentTime,
    setPlaybackSpeed
  } = useDemoControls();

  const {
    hotkeysEnabled,
    triggerAttentionDip,
    triggerParticipationBoost,
    triggerEmotionShift,
    triggerAlert
  } = useDemoHotkeys();

  const scenario = useDemoStore((state) => state.scenario);

  // Hotkey event handlers
  useEffect(() => {
    if (!hotkeysEnabled) return;

    const handleKeyPress = (event: KeyboardEvent) => {
      // Only trigger if Ctrl+Shift is pressed
      if (!event.ctrlKey || !event.shiftKey) return;

      switch (event.key.toLowerCase()) {
        case 'a':
          event.preventDefault();
          triggerAttentionDip();
          console.log('🔥 Demo: Attention dip triggered');
          break;
        case 'p':
          event.preventDefault();
          triggerParticipationBoost();
          console.log('🚀 Demo: Participation boost triggered');
          break;
        case 's':
          event.preventDefault();
          triggerEmotionShift('confused');
          console.log('😕 Demo: Emotion shift to confused');
          break;
        case 'h':
          event.preventDefault();
          triggerEmotionShift('happy');
          console.log('😊 Demo: Emotion shift to happy');
          break;
        case 'f':
          event.preventDefault();
          triggerEmotionShift('focused');
          console.log('🎯 Demo: Emotion shift to focused');
          break;
        case 'r':
          event.preventDefault();
          reset();
          console.log('🔄 Demo: Reset triggered');
          break;
        case 'd':
          event.preventDefault();
          triggerAlert('attention');
          console.log('⚠️ Demo: Alert triggered');
          break;
        default:
          break;
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [hotkeysEnabled, triggerAttentionDip, triggerParticipationBoost, triggerEmotionShift, triggerAlert, reset]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const speedOptions = [0.5, 1, 2, 4];

  if (!visible) {
    return (
      <div className="fixed bottom-4 left-4 z-50">
        <div className="text-xs text-gray-500 bg-black/20 backdrop-blur-sm rounded px-2 py-1">
          Demo: Ctrl+Shift+H for hotkeys
        </div>
      </div>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 100 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 100 }}
        transition={{ duration: 0.3 }}
        className={clsx(
          'fixed bottom-4 left-4 right-4 z-50',
          'glass-effect-dark',
          'rounded-lg',
          'p-4',
          'border border-primary-500/30',
          'backdrop-blur-lg',
          'max-w-4xl mx-auto',
          className
        )}
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-white">
              {t('demo.controls.title', 'Demo Controls')}
            </h3>
            <p className="text-sm text-gray-400">
              {scenario?.name || 'No scenario loaded'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Hotkeys:</span>
            <div className={clsx(
              'w-3 h-3 rounded-full',
              hotkeysEnabled ? 'bg-emerald-500' : 'bg-red-500'
            )} />
          </div>
        </div>

        {/* Playback Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              {t('demo.controls.playback', 'Playback')}
            </label>
            <div className="flex gap-2">
              <NeonButton
                size="sm"
                variant={isPlaying ? 'secondary' : 'primary'}
                onClick={isPlaying ? pause : play}
              >
                {isPlaying ? (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                )}
                {isPlaying ? t('demo.controls.pause') : t('demo.controls.play')}
              </NeonButton>
              <NeonButton size="sm" variant="secondary" onClick={reset}>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                {t('demo.controls.reset')}
              </NeonButton>
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">
              {t('demo.speed', 'Speed')}
            </label>
            <div className="flex gap-1">
              {speedOptions.map((speed) => (
                <button
                  key={speed}
                  onClick={() => setPlaybackSpeed(speed)}
                  className={clsx(
                    'px-2 py-1 text-xs rounded',
                    'transition-colors',
                    playbackSpeed === speed
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  )}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">
              {t('demo.timeline', 'Timeline')}
            </label>
            <div className="text-sm text-white">
              {formatTime(currentTime)} / {formatTime(scenario?.duration || 0)}
            </div>
            <input
              type="range"
              min={0}
              max={scenario?.duration || 0}
              value={currentTime}
              onChange={(e) => setCurrentTime(Number(e.target.value))}
              className="w-full mt-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>

        {/* Hotkey Actions */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <NeonButton
            size="sm"
            variant="accent"
            onClick={triggerAttentionDip}
            className="text-xs"
          >
            <span className="mr-1">Ctrl+Shift+A</span>
            Attention ↓
          </NeonButton>
          <NeonButton
            size="sm"
            variant="accent"
            onClick={triggerParticipationBoost}
            className="text-xs"
          >
            <span className="mr-1">Ctrl+Shift+P</span>
            Participation ↑
          </NeonButton>
          <NeonButton
            size="sm"
            variant="accent"
            onClick={() => triggerEmotionShift('confused')}
            className="text-xs"
          >
            <span className="mr-1">Ctrl+Shift+S</span>
            Confused 😕
          </NeonButton>
          <NeonButton
            size="sm"
            variant="accent"
            onClick={() => triggerAlert('attention')}
            className="text-xs"
          >
            <span className="mr-1">Ctrl+Shift+D</span>
            Alert ⚠️
          </NeonButton>
        </div>

        {/* Events Timeline */}
        {scenario && (
          <div className="mt-4 pt-4 border-t border-gray-600/30">
            <label className="block text-sm text-gray-400 mb-2">
              {t('demo.events', 'Events')}
            </label>
            <div className="flex gap-1 overflow-x-auto">
              {scenario.events.map((event, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentTime(event.timestamp)}
                  className={clsx(
                    'flex-shrink-0 px-2 py-1 text-xs rounded',
                    'transition-colors',
                    Math.abs(currentTime - event.timestamp) < 5
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  )}
                  title={event.description}
                >
                  {formatTime(event.timestamp)}
                </button>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

DemoControls.displayName = 'DemoControls';
