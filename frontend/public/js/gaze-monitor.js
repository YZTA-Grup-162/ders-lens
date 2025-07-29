/**
 * DersLens Gaze Detection Integration
 * Frontend JavaScript for real-time classroom attention monitoring
 */

class DersLensGazeMonitor {
    constructor(apiBaseUrl = '/api/gaze') {
        this.apiUrl = apiBaseUrl;
        this.isMonitoring = false;
        this.currentStream = null;
        this.stats = {
            totalFrames: 0,
            successfulDetections: 0,
            averageAttentionScore: 0
        };
        
        // Initialize UI elements
        this.initializeUI();
        
        console.log('🎯 DersLens Gaze Monitor initialized');
    }
    
    initializeUI() {
        // Create monitoring UI if not exists
        if (!document.getElementById('gaze-monitor-panel')) {
            this.createMonitoringPanel();
        }
        
        // Bind event listeners
        this.bindEventListeners();
    }
    
    createMonitoringPanel() {
        const panel = document.createElement('div');
        panel.id = 'gaze-monitor-panel';
        panel.className = 'gaze-monitor-panel';
        panel.innerHTML = `
            <div class="gaze-header">
                <h3>🎯 Classroom Attention Monitor</h3>
                <div class="gaze-status">
                    <span id="gaze-status-indicator" class="status-indicator">⚪</span>
                    <span id="gaze-status-text">Ready</span>
                </div>
            </div>
            
            <div class="gaze-controls">
                <button id="start-monitoring-btn" class="btn btn-primary">Start Monitoring</button>
                <button id="stop-monitoring-btn" class="btn btn-secondary" disabled>Stop Monitoring</button>
                <button id="capture-analysis-btn" class="btn btn-success">Analyze Current Frame</button>
            </div>
            
            <div class="gaze-stats">
                <div class="stat-item">
                    <span class="stat-label">Overall Attention:</span>
                    <span id="overall-attention" class="stat-value">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Students Detected:</span>
                    <span id="students-count" class="stat-value">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Focused Students:</span>
                    <span id="focused-count" class="stat-value">--</span>
                </div>
            </div>
            
            <div class="gaze-visualization">
                <canvas id="gaze-canvas" width="640" height="480"></canvas>
                <video id="gaze-video" autoplay muted style="display: none;"></video>
            </div>
            
            <div class="gaze-insights">
                <h4>📊 Real-time Insights</h4>
                <div id="attention-insights" class="insights-container">
                    <p>Start monitoring to see real-time attention analysis...</p>
                </div>
            </div>
        `;
        
        // Add CSS styles
        this.addStyles();
        
        // Append to body or target container
        const container = document.getElementById('monitoring-container') || document.body;
        container.appendChild(panel);
    }
    
    addStyles() {
        if (document.getElementById('gaze-monitor-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'gaze-monitor-styles';
        styles.textContent = `
            .gaze-monitor-panel {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                padding: 20px;
                margin: 20px;
                max-width: 800px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .gaze-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
            }
            
            .gaze-header h3 {
                margin: 0;
                color: #2c3e50;
            }
            
            .gaze-status {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .status-indicator {
                font-size: 12px;
                animation: pulse 2s infinite;
            }
            
            .gaze-controls {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .btn-primary { background: #3498db; color: white; }
            .btn-primary:hover { background: #2980b9; }
            .btn-secondary { background: #95a5a6; color: white; }
            .btn-secondary:hover { background: #7f8c8d; }
            .btn-success { background: #27ae60; color: white; }
            .btn-success:hover { background: #229954; }
            .btn:disabled { opacity: 0.6; cursor: not-allowed; }
            
            .gaze-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .stat-item {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            
            .stat-label {
                display: block;
                font-size: 12px;
                color: #7f8c8d;
                margin-bottom: 5px;
            }
            
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
            
            .gaze-visualization {
                margin-bottom: 20px;
                text-align: center;
            }
            
            #gaze-canvas {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                max-width: 100%;
                height: auto;
            }
            
            .gaze-insights {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
            }
            
            .insights-container {
                margin-top: 10px;
            }
            
            .attention-excellent { color: #27ae60; }
            .attention-good { color: #f39c12; }
            .attention-moderate { color: #e67e22; }
            .attention-poor { color: #e74c3c; }
            
            @keyframes pulse {
                0% { opacity: 0.5; }
                50% { opacity: 1; }
                100% { opacity: 0.5; }
            }
        `;
        
        document.head.appendChild(styles);
    }
    
    bindEventListeners() {
        document.getElementById('start-monitoring-btn')?.addEventListener('click', () => this.startMonitoring());
        document.getElementById('stop-monitoring-btn')?.addEventListener('click', () => this.stopMonitoring());
        document.getElementById('capture-analysis-btn')?.addEventListener('click', () => this.captureAndAnalyze());
    }
    
    updateStatus(status, text) {
        const indicator = document.getElementById('gaze-status-indicator');
        const statusText = document.getElementById('gaze-status-text');
        
        const statusMap = {
            'ready': { emoji: '⚪', color: '#95a5a6' },
            'monitoring': { emoji: '🟢', color: '#27ae60' },
            'processing': { emoji: '🟡', color: '#f39c12' },
            'error': { emoji: '🔴', color: '#e74c3c' }
        };
        
        if (indicator && statusText) {
            const config = statusMap[status] || statusMap['ready'];
            indicator.textContent = config.emoji;
            indicator.style.color = config.color;
            statusText.textContent = text;
        }
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/status`);
            const data = await response.json();
            
            if (data.status === 'ready' && data.model_loaded) {
                return { ready: true, message: 'System ready' };
            } else if (!data.model_loaded) {
                return { ready: false, message: 'Gaze model not loaded' };
            } else {
                return { ready: false, message: data.message || 'System not ready' };
            }
        } catch (error) {
            return { ready: false, message: `Connection error: ${error.message}` };
        }
    }
    
    async startMonitoring() {
        const status = await this.checkSystemStatus();
        if (!status.ready) {
            alert(`Cannot start monitoring: ${status.message}`);
            return;
        }
        
        try {
            // Get user media
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            const video = document.getElementById('gaze-video');
            video.srcObject = stream;
            this.currentStream = stream;
            
            this.isMonitoring = true;
            this.updateStatus('monitoring', 'Monitoring active');
            
            // Update UI
            document.getElementById('start-monitoring-btn').disabled = true;
            document.getElementById('stop-monitoring-btn').disabled = false;
            
            // Start processing frames
            this.processVideoFrames();
            
            console.log('Monitoring started');
            
        } catch (error) {
            console.error('Failed to start monitoring:', error);
            this.updateStatus('error', 'Failed to access camera');
            alert(`Failed to start monitoring: ${error.message}`);
        }
    }
    
    stopMonitoring() {
        this.isMonitoring = false;
        
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }
        
        this.updateStatus('ready', 'Monitoring stopped');
        
        // Update UI
        document.getElementById('start-monitoring-btn').disabled = false;
        document.getElementById('stop-monitoring-btn').disabled = true;
        
        console.log('⏹️ Monitoring stopped');
    }
    
    async processVideoFrames() {
        const video = document.getElementById('gaze-video');
        const canvas = document.getElementById('gaze-canvas');
        const ctx = canvas.getContext('2d');
        
        const processFrame = async () => {
            if (!this.isMonitoring) return;
            
            try {
                // Draw video frame to canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convert canvas to blob and analyze
                canvas.toBlob(async (blob) => {
                    if (blob) {
                        const results = await this.analyzeFrame(blob);
                        if (results) {
                            this.updateVisualization(results);
                            this.updateStats(results);
                        }
                    }
                }, 'image/jpeg', 0.8);
                
                // Continue processing
                setTimeout(processFrame, 2000); // Process every 2 seconds
                
            } catch (error) {
                console.error('Frame processing error:', error);
            }
        };
        
        // Wait for video to be ready
        video.addEventListener('loadeddata', processFrame, { once: true });
    }
    
    async analyzeFrame(imageBlob) {
        try {
            this.updateStatus('processing', 'Analyzing...');
            
            const formData = new FormData();
            formData.append('file', imageBlob, 'frame.jpg');
            
            const response = await fetch(`${this.apiUrl}/analyze_classroom`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }
            
            const results = await response.json();
            this.updateStatus('monitoring', 'Monitoring active');
            
            return results;
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.updateStatus('error', 'Analysis failed');
            return null;
        }
    }
    
    async captureAndAnalyze() {
        const canvas = document.getElementById('gaze-canvas');
        
        try {
            // If monitoring, use current frame; otherwise, capture from camera
            if (!this.isMonitoring) {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const video = document.createElement('video');
                video.srcObject = stream;
                video.play();
                
                video.addEventListener('loadeddata', async () => {
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    stream.getTracks().forEach(track => track.stop());
                    
                    canvas.toBlob(async (blob) => {
                        if (blob) {
                            const results = await this.analyzeFrame(blob);
                            if (results) {
                                this.updateVisualization(results);
                                this.updateStats(results);
                                this.showDetailedAnalysis(results);
                            }
                        }
                    }, 'image/jpeg', 0.8);
                });
            } else {
                // Use current monitoring frame
                canvas.toBlob(async (blob) => {
                    if (blob) {
                        const results = await this.analyzeFrame(blob);
                        if (results) {
                            this.showDetailedAnalysis(results);
                        }
                    }
                }, 'image/jpeg', 0.8);
            }
            
        } catch (error) {
            console.error('Capture error:', error);
            alert(`Failed to capture: ${error.message}`);
        }
    }
    
    updateVisualization(results) {
        const canvas = document.getElementById('gaze-canvas');
        const ctx = canvas.getContext('2d');
        
        // Draw attention zones on faces
        if (results.individual_students) {
            results.individual_students.forEach(student => {
                const [x, y, w, h] = student.bbox;
                const zone = student.attention_zone;
                
                // Scale coordinates to canvas size
                const scaleX = canvas.width / canvas.width; // Adjust based on actual video dimensions
                const scaleY = canvas.height / canvas.height;
                
                const canvasX = x * scaleX;
                const canvasY = y * scaleY;
                const canvasW = w * scaleX;
                const canvasH = h * scaleY;
                
                // Color mapping
                const colors = {
                    'focused': '#27ae60',
                    'attentive': '#f39c12',
                    'distracted': '#e67e22',
                    'off_task': '#e74c3c'
                };
                
                ctx.strokeStyle = colors[zone] || '#95a5a6';
                ctx.lineWidth = 3;
                ctx.strokeRect(canvasX, canvasY, canvasW, canvasH);
                
                // Draw label
                ctx.fillStyle = colors[zone] || '#95a5a6';
                ctx.font = '12px Arial';
                ctx.fillText(zone, canvasX, canvasY - 5);
            });
        }
    }
    
    updateStats(results) {
        if (results.attention_analysis) {
            const analysis = results.attention_analysis;
            
            // Update stat displays
            document.getElementById('overall-attention').textContent = 
                `${analysis.overall_score}% ${analysis.attention_emoji}`;
            document.getElementById('students-count').textContent = results.total_students;
            document.getElementById('focused-count').textContent = 
                analysis.distribution.focused.count;
            
            // Update insights
            this.updateInsights(results);
        }
    }
    
    updateInsights(results) {
        const insightsContainer = document.getElementById('attention-insights');
        const analysis = results.attention_analysis;
        
        let insights = `
            <div class="attention-overview">
                <h5>Current Classroom Status: ${analysis.attention_level.toUpperCase()} ${analysis.attention_emoji}</h5>
                <p><strong>Attention Score:</strong> ${analysis.overall_score}%</p>
            </div>
            
            <div class="attention-breakdown">
                <h6>Student Distribution:</h6>
                <ul>
                    <li>🎯 Focused: ${analysis.distribution.focused.count} (${analysis.distribution.focused.percentage}%)</li>
                    <li>👀 Attentive: ${analysis.distribution.attentive.count} (${analysis.distribution.attentive.percentage}%)</li>
                    <li>😐 Distracted: ${analysis.distribution.distracted.count} (${analysis.distribution.distracted.percentage}%)</li>
                    <li>😴 Off-task: ${analysis.distribution.off_task.count} (${analysis.distribution.off_task.percentage}%)</li>
                </ul>
            </div>
        `;
        
        if (results.recommendations && results.recommendations.length > 0) {
            insights += `
                <div class="recommendations">
                    <h6>💡 Recommendations:</h6>
                    <ul>
                        ${results.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        insightsContainer.innerHTML = insights;
    }
    
    showDetailedAnalysis(results) {
        // Create detailed analysis modal/panel
        const analysis = JSON.stringify(results, null, 2);
        console.log('📊 Detailed Analysis:', results);
        
        // You can implement a modal here or update a dedicated analysis panel
        alert(`Analysis Complete!\n\nTotal Students: ${results.total_students}\nAttention Score: ${results.attention_analysis?.overall_score || 'N/A'}%\n\nCheck console for detailed results.`);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dersLensGazeMonitor = new DersLensGazeMonitor();
    console.log('🎯 DersLens Gaze Monitor ready!');
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DersLensGazeMonitor;
}
