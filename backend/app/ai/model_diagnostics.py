"""
Model Diagnostic Tool for DersLens
Analyzes and reports on model performance and availability
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ModelDiagnostics:
    """
    Comprehensive model diagnostics and performance analysis
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.diagnostic_results = {}
        
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics on all models"""
        logger.info("🔍 Starting comprehensive model diagnostics...")
        
        results = {
            'timestamp': time.time(),
            'model_availability': self._check_model_availability(),
            'performance_analysis': self._analyze_performance(),
            'integration_status': self._check_integration_status(),
            'recommendations': self._generate_recommendations()
        }
        
        self.diagnostic_results = results
        self._generate_report()
        
        return results
    
    def _check_model_availability(self) -> Dict[str, Any]:
        """Check which models are available and properly configured"""
        availability = {
            'mpiigaze': self._check_mpiigaze(),
            'mendeley_ensemble': self._check_mendeley(),
            'daisee': self._check_daisee(), 
            'fer2013': self._check_fer2013(),
            'onnx_models': self._check_onnx(),
            'sklearn_models': self._check_sklearn()
        }
        
        total_available = sum(1 for status in availability.values() if status['available'])
        
        return {
            'models': availability,
            'total_available': total_available,
            'total_expected': len(availability),
            'availability_score': total_available / len(availability)
        }
    
    def _check_mpiigaze(self) -> Dict[str, Any]:
        """Check MPIIGaze model status"""
        model_path = self.base_dir / "models_mpiigaze" / "mpiigaze_best.pth"
        
        result = {
            'available': model_path.exists(),
            'path': str(model_path),
            'expected_accuracy': '3.39° MAE',
            'model_type': 'Gaze Direction',
            'priority': 'High (Best for gaze tracking)'
        }
        
        if result['available']:
            try:
                import torch
                model = torch.load(model_path, map_location='cpu')
                result['loadable'] = True
                result['model_size'] = model_path.stat().st_size / (1024*1024)  # MB
                result['status'] = 'Ready to use'
            except Exception as e:
                result['loadable'] = False
                result['error'] = str(e)
                result['status'] = 'Found but not loadable'
        else:
            result['status'] = 'Not found'
            result['loadable'] = False
        
        return result
    
    def _check_mendeley(self) -> Dict[str, Any]:
        """Check Mendeley ensemble models"""
        mendeley_dir = self.base_dir / "models_mendeley"
        
        expected_files = [
            'mendeley_gradient_boosting.pkl',
            'mendeley_random_forest.pkl',
            'mendeley_logistic_regression.pkl',
            'mendeley_scaler.pkl',
            'mendeley_nn_best.pth'
        ]
        
        found_files = []
        for file in expected_files:
            file_path = mendeley_dir / file
            if file_path.exists():
                found_files.append(file)
        
        result = {
            'available': len(found_files) >= 3,  # Need at least 3 models
            'path': str(mendeley_dir),
            'found_files': found_files,
            'missing_files': [f for f in expected_files if f not in found_files],
            'expected_accuracy': '87% (Ensemble)',
            'model_type': 'Attention Classification',
            'priority': 'High (Best overall accuracy)'
        }
        
        if result['available']:
            try:
                import pickle

                # Test loading one model
                test_file = mendeley_dir / found_files[0]
                with open(test_file, 'rb') as f:
                    pickle.load(f)
                result['loadable'] = True
                result['status'] = f"Ready ({len(found_files)}/{len(expected_files)} files)"
            except Exception as e:
                result['loadable'] = False
                result['error'] = str(e)
                result['status'] = 'Found but not loadable'
        else:
            result['status'] = 'Insufficient models found'
            result['loadable'] = False
        
        return result
    
    def _check_daisee(self) -> Dict[str, Any]:
        """Check DAiSEE model"""
        model_path = self.base_dir / "models" / "daisee_emotional_model_best.pth"
        
        result = {
            'available': model_path.exists(),
            'path': str(model_path),
            'expected_accuracy': '78% (Emotion-based)',
            'model_type': 'Emotion + Attention',
            'priority': 'Medium (Specialized for emotions)'
        }
        
        if result['available']:
            try:
                import torch
                model = torch.load(model_path, map_location='cpu')
                result['loadable'] = True
                result['model_size'] = model_path.stat().st_size / (1024*1024)
                result['status'] = 'Ready to use'
            except Exception as e:
                result['loadable'] = False
                result['error'] = str(e)
                result['status'] = 'Found but not loadable'
        else:
            result['status'] = 'Not found'
            result['loadable'] = False
        
        return result
    
    def _check_fer2013(self) -> Dict[str, Any]:
        """Check FER2013 emotion models"""
        fer_dir = self.base_dir / "models_fer2013"
        
        result = {
            'available': fer_dir.exists(),
            'path': str(fer_dir),
            'expected_accuracy': '72% (Emotion)',
            'model_type': 'Emotion Classification',
            'priority': 'Low (Backup emotion detection)'
        }
        
        if result['available']:
            model_files = list(fer_dir.glob("*.pth"))
            result['found_files'] = [f.name for f in model_files]
            result['loadable'] = len(model_files) > 0
            result['status'] = f"Found {len(model_files)} model files"
        else:
            result['status'] = 'Directory not found'
            result['loadable'] = False
        
        return result
    
    def _check_onnx(self) -> Dict[str, Any]:
        """Check ONNX models"""
        onnx_dir = self.base_dir / "models" / "onnx"
        
        result = {
            'available': onnx_dir.exists(),
            'path': str(onnx_dir),
            'expected_accuracy': '70% (ONNX)',
            'model_type': 'Cross-platform models',
            'priority': 'Low (Fallback)'
        }
        
        if result['available']:
            onnx_files = list(onnx_dir.glob("*.onnx"))
            result['found_files'] = [f.name for f in onnx_files]
            
            try:
                import onnxruntime
                result['runtime_available'] = True
                result['loadable'] = len(onnx_files) > 0
                result['status'] = f"ONNX Runtime available, {len(onnx_files)} models"
            except ImportError:
                result['runtime_available'] = False
                result['loadable'] = False
                result['status'] = 'ONNX Runtime not installed'
        else:
            result['status'] = 'Directory not found'
            result['loadable'] = False
        
        return result
    
    def _check_sklearn(self) -> Dict[str, Any]:
        """Check scikit-learn models"""
        models_dir = self.base_dir / "models"
        
        expected_files = [
            'local_attention_model_random_forest.pkl',
            'local_attention_model_gradient_boosting.pkl',
            'local_scaler_random_forest.pkl',
            'local_scaler_gradient_boosting.pkl'
        ]
        
        found_files = []
        for file in expected_files:
            file_path = models_dir / file
            if file_path.exists():
                found_files.append(file)
        
        result = {
            'available': len(found_files) >= 2,
            'path': str(models_dir),
            'found_files': found_files,
            'missing_files': [f for f in expected_files if f not in found_files],
            'expected_accuracy': '65% (Basic ML)',
            'model_type': 'Traditional ML',
            'priority': 'Low (Basic fallback)'
        }
        
        if result['available']:
            try:
                import pickle

                import sklearn

                # Test loading one model
                test_file = models_dir / found_files[0]
                with open(test_file, 'rb') as f:
                    pickle.load(f)
                result['loadable'] = True
                result['sklearn_version'] = sklearn.__version__
                result['status'] = f"Sklearn available, {len(found_files)} files"
            except Exception as e:
                result['loadable'] = False
                result['error'] = str(e)
                result['status'] = 'Found but not loadable'
        else:
            result['status'] = 'Insufficient models'
            result['loadable'] = False
        
        return result
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze expected performance based on available models"""
        availability = self._check_model_availability()
        
        performance_ranking = []
        
        for model_name, info in availability['models'].items():
            if info['available'] and info['loadable']:
                performance_ranking.append({
                    'model': model_name,
                    'priority': info['priority'],
                    'accuracy': info['expected_accuracy'],
                    'type': info['model_type']
                })
        
        # Calculate overall system performance
        if any(m['model'] == 'mpiigaze' for m in performance_ranking):
            expected_performance = 'Excellent (MPIIGaze available)'
            performance_score = 0.95
        elif any(m['model'] == 'mendeley_ensemble' for m in performance_ranking):
            expected_performance = 'Very Good (Mendeley ensemble available)'
            performance_score = 0.87
        elif any(m['model'] == 'daisee' for m in performance_ranking):
            expected_performance = 'Good (DAiSEE available)'
            performance_score = 0.78
        elif any(m['model'] == 'sklearn_models' for m in performance_ranking):
            expected_performance = 'Basic (Sklearn fallback)'
            performance_score = 0.65
        else:
            expected_performance = 'Poor (Heuristic only)'
            performance_score = 0.4
        
        return {
            'expected_performance': expected_performance,
            'performance_score': performance_score,
            'available_models': performance_ranking,
            'recommended_model': performance_ranking[0] if performance_ranking else None
        }
    
    def _check_integration_status(self) -> Dict[str, Any]:
        """Check integration status of various components"""
        integration_checks = {
            'enhanced_detector': self._check_enhanced_detector(),
            'unified_manager': self._check_unified_manager(),
            'attention_detector': self._check_attention_detector(),
            'dependencies': self._check_dependencies()
        }
        
        return integration_checks
    
    def _check_enhanced_detector(self) -> Dict[str, Any]:
        """Check if enhanced detector module is properly integrated"""
        try:
            from .enhanced_dataset_integration import EnhancedAttentionDetector
            detector = EnhancedAttentionDetector()
            return {
                'available': True,
                'status': 'Properly integrated',
                'models_loaded': len(detector.models)
            }
        except Exception as e:
            return {
                'available': False,
                'status': f'Integration failed: {e}',
                'error': str(e)
            }
    
    def _check_unified_manager(self) -> Dict[str, Any]:
        """Check unified model manager"""
        try:
            from .unified_model_manager import UnifiedModelManager
            manager = UnifiedModelManager(str(self.base_dir))
            status = manager.get_model_status()
            return {
                'available': True,
                'status': 'Unified manager working',
                'loaded_models': status['loaded_models'],
                'total_models': status['total_models']
            }
        except Exception as e:
            return {
                'available': False,
                'status': f'Manager failed: {e}',
                'error': str(e)
            }
    
    def _check_attention_detector(self) -> Dict[str, Any]:
        """Check basic attention detector"""
        try:
            # Create test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Try importing and using attention detector
            from .attention_detector import AttentionDetector
            detector = AttentionDetector()
            result = detector.analyze_attention(test_frame)
            
            return {
                'available': True,
                'status': 'Basic detector working',
                'enhanced_available': detector.use_enhanced
            }
        except Exception as e:
            return {
                'available': False,
                'status': f'Detector failed: {e}',
                'error': str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        dependencies = {
            'opencv': self._check_dependency('cv2'),
            'numpy': self._check_dependency('numpy'),
            'mediapipe': self._check_dependency('mediapipe'),
            'torch': self._check_dependency('torch'),
            'sklearn': self._check_dependency('sklearn'),
            'onnxruntime': self._check_dependency('onnxruntime')
        }
        
        critical_missing = [name for name, status in dependencies.items() 
                          if not status['available'] and name in ['opencv', 'numpy', 'mediapipe']]
        
        return {
            'dependencies': dependencies,
            'critical_missing': critical_missing,
            'all_critical_available': len(critical_missing) == 0
        }
    
    def _check_dependency(self, module_name: str) -> Dict[str, Any]:
        """Check if a dependency is available"""
        try:
            module = __import__(module_name)
            return {
                'available': True,
                'version': getattr(module, '__version__', 'Unknown'),
                'status': 'Available'
            }
        except ImportError:
            return {
                'available': False,
                'status': 'Not installed'
            }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on diagnostic results"""
        recommendations = []
        
        availability = self._check_model_availability()
        
        # Check for missing high-priority models
        if not availability['models']['mpiigaze']['available']:
            recommendations.append(
                "🎯 CRITICAL: MPIIGaze model missing. This is your best model (3.39° MAE). "
                "Copy mpiigaze_best.pth to models_mpiigaze/ directory."
            )
        
        if not availability['models']['mendeley_ensemble']['available']:
            recommendations.append(
                "⚠️ HIGH: Mendeley ensemble models missing. These provide 87% accuracy. "
                "Check models_mendeley/ directory for .pkl files."
            )
        
        # Check integration issues
        integration = self._check_integration_status()
        
        if not integration['enhanced_detector']['available']:
            recommendations.append(
                "🔧 INTEGRATION: Enhanced detector not working. "
                "Check enhanced_dataset_integration.py module."
            )
        
        if not integration['dependencies']['all_critical_available']:
            missing = integration['dependencies']['critical_missing']
            recommendations.append(
                f"📦 DEPENDENCIES: Critical dependencies missing: {', '.join(missing)}. "
                "Install with: pip install opencv-python mediapipe numpy"
            )
        
        # Performance recommendations
        performance = self._analyze_performance()
        
        if performance['performance_score'] < 0.8:
            recommendations.append(
                "📈 PERFORMANCE: System performance below optimal. "
                "Ensure high-accuracy models (MPIIGaze, Mendeley) are available."
            )
        
        # Model-specific recommendations
        if availability['models']['onnx_models']['available'] and not availability['models']['onnx_models']['loadable']:
            recommendations.append(
                "🔧 ONNX: ONNX models found but ONNX Runtime not installed. "
                "Install with: pip install onnxruntime"
            )
        
        return recommendations
    
    def _generate_report(self):
        """Generate comprehensive diagnostic report"""
        if not self.diagnostic_results:
            return
        
        results = self.diagnostic_results
        
        print("\\n" + "="*80)
        print("🔍 DERSLENS MODEL DIAGNOSTICS REPORT")
        print("="*80)
        
        # Model Availability Summary
        availability = results['model_availability']
        print(f"\\n📊 MODEL AVAILABILITY: {availability['total_available']}/{availability['total_expected']} models available")
        print(f"   Overall Availability Score: {availability['availability_score']:.2%}")
        
        for model_name, info in availability['models'].items():
            status_icon = "✅" if info['available'] and info.get('loadable', True) else "❌"
            print(f"   {status_icon} {model_name:20} - {info['status']}")
            if info['available']:
                print(f"      └─ {info['expected_accuracy']} | {info['priority']}")
        
        # Performance Analysis
        performance = results['performance_analysis']
        print(f"\\n🎯 EXPECTED PERFORMANCE: {performance['expected_performance']}")
        print(f"   Performance Score: {performance['performance_score']:.2%}")
        
        if performance['recommended_model']:
            rec = performance['recommended_model']
            print(f"   Recommended Model: {rec['model']} ({rec['accuracy']})")
        
        # Integration Status
        integration = results['integration_status']
        print("\\n🔧 INTEGRATION STATUS:")
        for component, status in integration.items():
            if component == 'dependencies':
                continue
            icon = "✅" if status['available'] else "❌"
            print(f"   {icon} {component:20} - {status['status']}")
        
        # Dependencies
        deps = integration['dependencies']
        print(f"\\n📦 DEPENDENCIES: {len([d for d in deps['dependencies'].values() if d['available']])}/{len(deps['dependencies'])} available")
        for dep_name, dep_info in deps['dependencies'].items():
            icon = "✅" if dep_info['available'] else "❌"
            version = f" v{dep_info['version']}" if dep_info['available'] else ""
            print(f"   {icon} {dep_name:15}{version}")
        
        # Recommendations
        recommendations = results['recommendations']
        if recommendations:
            print("\\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("\\n✅ ALL SYSTEMS OPTIMAL - No recommendations needed!")
        
        print("\\n" + "="*80)
        print(f"📅 Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\\n")

# Diagnostic CLI function
def run_diagnostics():
    """Run diagnostics from command line"""
    import sys

    # Get base directory from command line or use current directory
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    diagnostics = ModelDiagnostics(base_dir)
    results = diagnostics.run_full_diagnostics()
    
    # Return exit code based on results
    performance_score = results['performance_analysis']['performance_score']
    availability_score = results['model_availability']['availability_score']
    
    if performance_score >= 0.8 and availability_score >= 0.7:
        sys.exit(0)  # Success
    elif performance_score >= 0.6:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Critical issues

if __name__ == "__main__":
    run_diagnostics()
