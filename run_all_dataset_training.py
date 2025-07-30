"""
Master Script for All Dataset Training
Runs all specialized dataset training scripts and provides comprehensive results.
Each dataset is trained separately with optimized parameters for 80%+ validation accuracy.
"""
import os
import sys
import logging
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from thermal_management import ThermalManager
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('all_dataset_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
class DatasetTrainingManager:
    def __init__(self):
        self.training_scripts = {
            'daisee': {
                'script': 'daisee_attention_training.py',
                'description': 'DAiSEE Attention Detection',
                'type': 'image',
                'target_acc': 80.0,
                'model_dir': 'models_daisee'
            },
            'fer2013': {
                'script': 'fer2013_emotion_training.py',
                'description': 'FER2013 Emotion Recognition',
                'type': 'image',
                'target_acc': 80.0,
                'model_dir': 'models_fer2013'
            },
            'ravdess': {
                'script': 'ravdess_emotion_training.py',
                'description': 'RAVDESS Audio Emotion Recognition',
                'type': 'audio',
                'target_acc': 80.0,
                'model_dir': 'models_ravdess'
            },
            'mendeley': {
                'script': 'mendeley_attention_training.py',
                'description': 'Mendeley Attention Detection',
                'type': 'tabular',
                'target_acc': 80.0,
                'model_dir': 'models_mendeley'
            },
            'mpiigaze': {
                'script': 'mpiigaze_training.py',
                'description': 'MPIIGaze Gaze Direction',
                'type': 'image',
                'target_acc': 80.0,
                'model_dir': 'models_mpiigaze'
            }
        }
        self.results = {}
        self.start_time = None
        self.thermal_manager = ThermalManager()
        self.thermal_manager.start_monitoring()
    def check_dataset_availability(self):
        logger.info("🔍 Checking dataset availability...")
        available_datasets = {}
        dataset_paths = {
            'daisee': 'datasets/daisee/Labels/labels.csv',
            'fer2013': 'datasets/fer_2013/train',
            'ravdess': 'datasets/ravdess',
            'mendeley': 'datasets/mendeley/Students Attention Detection Dataset/attention_detection_dataset_v1.csv',
            'mpiigaze': 'datasets/MPIIGaze/Data'
        }
        for dataset_name, path in dataset_paths.items():
            if Path(path).exists():
                available_datasets[dataset_name] = self.training_scripts[dataset_name]
                logger.info(f"✅ {dataset_name}: Available")
            else:
                logger.warning(f"{dataset_name}: Not found at {path}")
        return available_datasets
    def run_training_script(self, dataset_name, script_info):
        logger.info(f"🚀 Starting {dataset_name} training...")
        logger.info(f"📋 Description: {script_info['description']}")
        logger.info(f"🎯 Target Accuracy: {script_info['target_acc']}%")
        self.thermal_manager.log_temperature()
        if not self.thermal_manager.is_safe_to_train():
            logger.error(f"System temperature unsafe or cannot be read. Skipping {dataset_name} training.")
            self.results[dataset_name] = {
                'status': 'skipped_thermal',
                'reason': 'unsafe_temperature_or_unreadable'
            }
            return False
        script_path = script_info['script']
        if not Path(script_path).exists():
            logger.error(f"Training script not found: {script_path}")
            return False
        try:
            start_time = time.time()
            venv_python = Path("venv/Scripts/python.exe")
            if not venv_python.exists():
                python_cmd = [sys.executable]
            else:
                python_cmd = [str(venv_python)]
            result = subprocess.run(
                python_cmd + [script_path],
                capture_output=True,
                text=True,
                timeout=7200
            )
            end_time = time.time()
            training_duration = end_time - start_time
            self.thermal_manager.log_temperature()
            if result.returncode == 0:
                logger.info(f"✅ {dataset_name} training completed successfully!")
                logger.info(f"⏱️ Training duration: {training_duration/60:.2f} minutes")
                model_dir = Path(script_info['model_dir'])
                if model_dir.exists():
                    results_file = model_dir / f"{dataset_name}_training_results.json"
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            training_results = json.load(f)
                        best_acc = training_results.get('best_val_acc', 0)
                        logger.info(f"🎯 Best validation accuracy: {best_acc:.2f}%")
                        self.results[dataset_name] = {
                            'status': 'success',
                            'best_val_acc': best_acc,
                            'training_duration': training_duration,
                            'target_met': best_acc >= script_info['target_acc'],
                            'results_file': str(results_file)
                        }
                    else:
                        logger.warning(f"⚠️ No results file found for {dataset_name}")
                        self.results[dataset_name] = {
                            'status': 'completed_no_results',
                            'training_duration': training_duration
                        }
                else:
                    logger.warning(f"⚠️ No model directory found for {dataset_name}")
                    self.results[dataset_name] = {
                        'status': 'completed_no_models',
                        'training_duration': training_duration
                    }
                return True
            else:
                logger.error(f"{dataset_name} training failed!")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                self.results[dataset_name] = {
                    'status': 'failed',
                    'error': result.stderr,
                    'training_duration': training_duration
                }
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"{dataset_name} training timed out after 2 hours")
            self.results[dataset_name] = {
                'status': 'timeout',
                'training_duration': 7200
            }
            return False
        except Exception as e:
            logger.error(f"{dataset_name} training error: {e}")
            self.results[dataset_name] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    def run_all_training(self):
        logger.info("🎯 Starting comprehensive dataset training...")
        self.start_time = time.time()
        available_datasets = self.check_dataset_availability()
        if not available_datasets:
            logger.error("No datasets available for training!")
            return
        logger.info(f"📊 Found {len(available_datasets)} available datasets")
        for dataset_name, script_info in available_datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 Training {dataset_name.upper()}")
            logger.info(f"{'='*60}")
            success = self.run_training_script(dataset_name, script_info)
            if success:
                logger.info(f"✅ {dataset_name} training completed")
            else:
                logger.error(f"{dataset_name} training failed")
            logger.info(f"{'='*60}\n")
        self.generate_training_report()
    def generate_training_report(self):
        logger.info("📊 Generating comprehensive training report...")
        end_time = time.time()
        total_duration = (end_time - self.start_time) if self.start_time is not None else 0
        report = {
            'training_summary': {
                'total_datasets': len(self.results),
                'successful_trainings': len([r for r in self.results.values() if r.get('status') == 'success']),
                'failed_trainings': len([r for r in self.results.values() if r.get('status') != 'success']),
                'total_duration_minutes': total_duration / 60,
                'timestamp': datetime.now().isoformat()
            },
            'dataset_results': self.results,
            'performance_analysis': self.analyze_performance()
        }
        report_file = Path("comprehensive_training_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.generate_summary_table()
        self.print_training_summary()
        logger.info(f"📄 Comprehensive report saved to: {report_file}")
    def analyze_performance(self):
        successful_results = [r for r in self.results.values() if r.get('status') == 'success']
        if not successful_results:
            return {'error': 'No successful trainings to analyze'}
        accuracies = [r.get('best_val_acc', 0) for r in successful_results]
        durations = [r.get('training_duration', 0) for r in successful_results]
        analysis = {
            'average_accuracy': sum(accuracies) / len(accuracies),
            'max_accuracy': max(accuracies),
            'min_accuracy': min(accuracies),
            'average_duration_minutes': sum(durations) / len(durations) / 60,
            'total_duration_hours': sum(durations) / 3600,
            'datasets_above_80_percent': len([acc for acc in accuracies if acc >= 80.0])
        }
        return analysis
    def generate_summary_table(self):
        logger.info("📋 Generating summary table...")
        summary_data = []
        for dataset_name, result in self.results.items():
            row = {
                'Dataset': dataset_name.upper(),
                'Status': result.get('status', 'unknown'),
                'Best Val Acc (%)': f"{result.get('best_val_acc', 0):.2f}" if result.get('best_val_acc') else 'N/A',
                'Duration (min)': f"{result.get('training_duration', 0)/60:.1f}" if result.get('training_duration') else 'N/A',
                'Target Met': '✅' if result.get('target_met', False) else '❌'
            }
            summary_data.append(row)
        df = pd.DataFrame(summary_data)
        summary_file = Path("training_summary_table.csv")
        df.to_csv(summary_file, index=False)
        logger.info(f"📊 Summary table saved to: {summary_file}")
        print("\n" + "="*80)
        print("TRAINING SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
    def print_training_summary(self):
        logger.info("\n" + "🎉 TRAINING COMPLETION SUMMARY")
        logger.info("="*50)
        successful = [name for name, result in self.results.items() if result.get('status') == 'success']
        failed = [name for name, result in self.results.items() if result.get('status') != 'success']
        logger.info(f"✅ Successful trainings: {len(successful)}")
        for dataset in successful:
            acc = self.results[dataset].get('best_val_acc', 0)
            logger.info(f"   - {dataset.upper()}: {acc:.2f}%")
        if failed:
            logger.info(f"Failed trainings: {len(failed)}")
            for dataset in failed:
                status = self.results[dataset].get('status', 'unknown')
                logger.info(f"   - {dataset.upper()}: {status}")
        successful_results = [r for r in self.results.values() if r.get('status') == 'success']
        if successful_results:
            avg_acc = sum(r.get('best_val_acc', 0) for r in successful_results) / len(successful_results)
            above_80 = len([r for r in successful_results if r.get('best_val_acc', 0) >= 80.0])
            logger.info(f"\n📊 Performance Summary:")
            logger.info(f"   - Average accuracy: {avg_acc:.2f}%")
            logger.info(f"   - Datasets above 80%: {above_80}/{len(successful_results)}")
        logger.info("="*50)
def main():
    logger.info("🚀 Starting comprehensive dataset training pipeline...")
    manager = DatasetTrainingManager()
    manager.run_all_training()
    logger.info("🎉 All dataset training completed!")
if __name__ == "__main__":
    main() 