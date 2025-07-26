import gc
import json
import logging
import platform
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
class ThermalManager:
    """
    Manages system thermal state during training or heavy computation.

    Monitors CPU and GPU temperatures, applies cooling strategies (such as reducing batch size, 
    reducing workers, adding delays, or lowering precision), and can trigger emergency cooling 
    to prevent hardware damage. Designed to be integrated with training loops and data loaders 
    to ensure safe operation under thermal constraints.
    """
    def __init__(self, max_temp=85, warning_temp=80, check_interval=5):
        self.max_temp = max_temp
        self.warning_temp = warning_temp
        self.check_interval = check_interval
        self.monitoring = False
        self.thermal_throttling = False
        self.cooling_strategies = {
            'reduce_batch_size': False,
            'reduce_workers': False,
            'add_delays': False,
            'reduce_precision': False,
            'emergency_stop': False
        }
        self.original_settings = {}
    def get_cpu_temperature(self):
        if platform.system() == "Windows":
            try:
                import wmi
                c = wmi.WMI(namespace="root\\wmi")
                temperature_info = c.MSAcpi_ThermalZoneTemperature()
                if temperature_info:
                    temp_kelvin = temperature_info[0].CurrentTemperature
                    return (temp_kelvin / 10.0) - 273.15
            except Exception as e:
                logger.warning(f"Error getting Windows CPU temperature: {e}")
            
            # Fallback for Windows if WMI fails
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                return 40 + (cpu_percent * 0.5)  # Estimate based on CPU usage
            except Exception as e:
                logger.warning(f"Error estimating CPU temperature: {e}")
                return None
                
        elif platform.system() == "Linux":
            try:
                thermal_zones = Path("/sys/class/thermal").glob("thermal_zone*/temp")
                for zone in thermal_zones:
                    try:
                        temp = int(zone.read_text()) / 1000.0
                        if 30 <= temp <= 120: 
                            return temp
                    except (ValueError, PermissionError, OSError) as e:
                        logger.debug(f"Error reading thermal zone {zone}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Error reading thermal zones: {e}")
            
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                return 40 + (cpu_percent * 0.3)  
            except Exception as e:
                logger.warning(f"Error estimating CPU temperature: {e}")
        
        return None  

    def get_gpu_temperature(self):
        if not torch.cuda.is_available():
            return None
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return None
    def get_system_stats(self):
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_temperature': self.get_cpu_temperature(),
            'gpu_temperature': self.get_gpu_temperature(),
        }
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        return stats
        return stats
        logger.info(f"🌡️ Applying cooling strategy: {strategy}")
        if strategy == 'reduce_batch_size':
            if dataloader_kwargs and 'batch_size' in dataloader_kwargs:
                original_batch = dataloader_kwargs['batch_size']
                new_batch = max(1, original_batch // 2)
                dataloader_kwargs['batch_size'] = new_batch
                logger.info(f"   Reduced batch size: {original_batch} → {new_batch}")
                return True
        elif strategy == 'reduce_workers':
            if dataloader_kwargs and 'num_workers' in dataloader_kwargs:
                original_workers = dataloader_kwargs['num_workers']
                new_workers = max(0, original_workers // 2)
                dataloader_kwargs['num_workers'] = new_workers
                logger.info(f"   Reduced workers: {original_workers} → {new_workers}")
                return True
        elif strategy == 'add_delays':
            logger.info("   Adding cooling delays between batches")
            return True
        elif strategy == 'reduce_precision':
            if model:
                if next(model.parameters()).dtype != torch.float16:
                    try:
                        model.half()
                        logger.info("   Converted model to half precision")
                        return True
                    except:
                        logger.warning("   Failed to convert to half precision")
        return False
    def emergency_cooling(self):
        logger.warning("🚨 EMERGENCY COOLING ACTIVATED!")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("   Waiting 30 seconds for system cooling...")
        time.sleep(30)
        current_temp = self.get_current_temperature()
        if current_temp and current_temp > self.max_temp:
            logger.error("   System still overheating after emergency cooling!")
            return False
        else:
            logger.info("   System temperature stabilized")
            return True
    def get_current_temperature(self):
        cpu_temp = self.get_cpu_temperature()
        gpu_temp = self.get_gpu_temperature()
        temps = [t for t in [cpu_temp, gpu_temp] if t is not None]
        return max(temps) if temps else None
    def check_thermal_state(self):
        current_temp = self.get_current_temperature()
        if current_temp is None:
            return "unknown", []
        self.temperature_history.append({
            'timestamp': time.time(),
            'temperature': current_temp
        })
        if len(self.temperature_history) > 60:
            self.temperature_history = self.temperature_history[-60:]
        if current_temp >= self.max_temp:
            return "critical", ["emergency_cooling", "reduce_batch_size", "add_delays"]
        elif current_temp >= self.warning_temp:
            return "warning", ["reduce_batch_size", "reduce_workers", "add_delays"]
        elif current_temp >= self.warning_temp - 5:
            return "elevated", ["reduce_workers"]
        else:
            return "normal", []
    def start_monitoring(self):
        self.monitoring = True
        def monitor_loop():
            while self.monitoring:
                try:
                    state, actions = self.check_thermal_state()
                    current_temp = self.get_current_temperature()
                    self.log_temperature()
                    if current_temp:
                        if state == "critical":
                            logger.error(f"🔥 CRITICAL TEMPERATURE: {current_temp}°C")
                            self.thermal_throttling = True
                        elif state == "warning":
                            logger.warning(f"⚠️ HIGH TEMPERATURE: {current_temp}°C")
                            self.thermal_throttling = True
                        elif state == "elevated":
                            logger.info(f"🌡️ Elevated temperature: {current_temp}°C")
                        else:
                            if self.thermal_throttling:
                                logger.info(f"✅ Temperature normalized: {current_temp}°C")
                                self.thermal_throttling = False
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Error in thermal monitoring: {e}")
                    time.sleep(self.check_interval)
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🌡️ Thermal monitoring started")
    def stop_monitoring(self):
        self.monitoring = False
        logger.info("🌡️ Thermal monitoring stopped")
    def save_thermal_log(self, filepath):
        thermal_log = {
            'temperature_history': self.temperature_history,
            'cooling_strategies_used': self.cooling_strategies,
            'max_temp_threshold': self.max_temp,
            'warning_temp_threshold': self.warning_temp
        }
        with open(filepath, 'w') as f:
            json.dump(thermal_log, f, indent=2)
    def is_safe_to_train(self):
        current_temp = self.get_current_temperature()
        if current_temp is None:
            logger.error("❌ Cannot determine system temperature, stopping training for safety")
            return False
        if current_temp >= self.max_temp:
            logger.error(f"🔥 Temperature too high: {current_temp}°C (max: {self.max_temp}°C)")
            return False
        elif current_temp >= self.warning_temp:
            logger.warning(f"⚠️ Temperature elevated: {current_temp}°C (warning: {self.warning_temp}°C)")
            return True
        else:
            return True
    def log_temperature(self):
        stats = self.get_system_stats()
        logger.info(f"[THERMAL LOG] {json.dumps(stats)}")
class CoolDataLoader:
    def __init__(self, dataset, thermal_manager, **kwargs):
        self.dataset = dataset
        self.thermal_manager = thermal_manager
        self.dataloader_kwargs = kwargs
        self.current_dataloader = None
        self.batch_delay = 0.0
        self._create_dataloader()
    def _create_dataloader(self):
        self.current_dataloader = DataLoader(self.dataset, **self.dataloader_kwargs)
    def __iter__(self):
        if self.current_dataloader is None:
            raise RuntimeError("DataLoader not initialized.")
        for batch_idx, batch in enumerate(self.current_dataloader):
            if self.thermal_manager.monitoring:
                state, actions = self.thermal_manager.check_thermal_state()
                if state == "critical":
                    logger.warning("🔥 Critical temperature - applying emergency cooling")
                    if not self.thermal_manager.emergency_cooling():
                        raise RuntimeError("System overheating - training stopped for safety")
                elif state == "warning" and not self.thermal_manager.thermal_throttling:
                    for action in actions:
                        if action == "reduce_batch_size":
                            old_batch = self.dataloader_kwargs.get('batch_size', 32)
                            new_batch = max(1, old_batch // 2)
                            if new_batch != old_batch:
                                self.dataloader_kwargs['batch_size'] = new_batch
                                self._create_dataloader()
                                logger.info(f"   Reduced batch size to {new_batch}")
                                break
                        elif action == "add_delays":
                            self.batch_delay = min(2.0, self.batch_delay + 0.1)
            yield batch
            if self.batch_delay > 0:
                time.sleep(self.batch_delay)
    def __len__(self):
        if self.current_dataloader is None:
            raise RuntimeError("DataLoader not initialized.")
        return len(self.current_dataloader)
def create_thermal_safe_training_config():
    return {
        'batch_size': 16,  # Smaller batch size
        'num_workers': 2,  # Fewer workers
        'pin_memory': False,  # Reduce memory pressure
        'persistent_workers': False,
        'prefetch_factor': 1,  # Reduce prefetching
        'thermal_check_interval': 5,  # Check every 5 seconds
        'max_temp': 85,  # Maximum temperature
        'warning_temp': 80,  # Warning temperature
        'enable_mixed_precision': True,  # Use AMP for efficiency
        'gradient_accumulation_steps': 2,  # Simulate larger batches
        'save_checkpoint_every': 5,  # Save more frequently
    }