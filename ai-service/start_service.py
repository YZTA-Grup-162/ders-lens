#!/usr/bin/env python3
"""
Ders Lens AI Servisi başlangıç betiği
Ortam sorunlarını kontrol eder ve detaylı hata raporu sunar
"""
import logging
import os
import sys
import warnings

# Ortam uyarılarını bastır (import öncesi)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*SymbolDatabase.GetPrototype.*")

# Mevcut dizini PYTHONPATH'e ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_and_fix_environment():
    """Ortam hatalarını kontrol et ve düzelt"""
    try:
        # Doğru dizinde olup olmadığımızı kontrol et
        if not os.path.exists('app.py'):
            logger.error("app.py bulunamadı. 'ai-service' dizininde çalışın.")
            return False
            
        # Kritik model dizinlerini kontrol et
        model_dirs = ['../models_mendeley', '../models_mpiigaze']
        for model_dir in model_dirs:
            if not os.path.exists(model_dir):
                logger.warning(f"Model dizini bulunamadı: {model_dir}")
        
        logger.info("Ortam kontrolü başarılı")
        return True
        
    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        return False

def start_ai_service():
    """Start the AI service with proper error handling"""
    try:
        logger.info("Ders Lens AI Servisi başlatılıyor...")
        
        # Import and start the app
        import uvicorn

        from app import app
        
        logger.info("AI Servisi başarıyla yüklendi")
        logger.info("Sunucu başlatılıyor: http://0.0.0.0:8003")
        
        # Start the server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8003, 
            log_config=None,
            reload=False  # Disable reload to prevent issues
        )
        
    except KeyboardInterrupt:
        logger.info("AI Servisi kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"AI Servisi başlatılamadı: {e}")
        logger.error("Önce 'python fix_startup.py' komutunu çalıştırın")
        return False
    
    return True

def main():
    """Main startup function"""
    logger.info("Ders Lens AI Servisi Başlatılıyor")
    logger.info("=" * 50)
    
    # Check environment
    if not check_and_fix_environment():
        logger.error("Environment check failed")
        return False
    
    # Start the service
    return start_ai_service()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
