import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Brain, 
  Users, 
  Lightbulb, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  TrendingUp,
  MessageSquare,
  Target,
  Gamepad2
} from 'lucide-react';

const GeminiAIDashboard = () => {
  const [classInsights, setClassInsights] = useState(null);
  const [studentAlerts, setStudentAlerts] = useState([]);
  const [adaptiveQuiz, setAdaptiveQuiz] = useState(null);
  const [showQuiz, setShowQuiz] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch real-time AI insights
  useEffect(() => {
    const fetchInsights = async () => {
      setIsLoading(true);
      try {
        const response = await fetch('/api/gemini/class-insights');
        const data = await response.json();
        setClassInsights(data);
      } catch (error) {
        console.error('Failed to fetch insights:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchInsights();
    const interval = setInterval(fetchInsights, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Generate adaptive quiz for disengaged students
  const generateAdaptiveQuiz = async (topic, difficulty, engagementLevel) => {
    try {
      const response = await fetch('/api/gemini/adaptive-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, difficulty, engagement_level: engagementLevel })
      });
      const quiz = await response.json();
      setAdaptiveQuiz(quiz);
      setShowQuiz(true);
    } catch (error) {
      console.error('Failed to generate quiz:', error);
    }
  };

  const getRiskBadgeColor = (riskLevel) => {
    switch (riskLevel) {
      case 'high': return 'destructive';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'secondary';
    }
  };

  const getEngagementIcon = (level) => {
    if (level >= 80) return <CheckCircle className="text-green-500" />;
    if (level >= 60) return <Clock className="text-yellow-500" />;
    return <AlertTriangle className="text-red-500" />;
  };

  return (
    <div className="p-6 space-y-6 bg-gradient-to-br from-purple-50 to-blue-50 min-h-screen">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="w-8 h-8 text-purple-600" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
            AI Teaching Assistant
          </h1>
        </div>
        <Badge variant="outline" className="text-sm">
          Powered by Gemini AI ✨
        </Badge>
      </div>

      {/* Class Health Overview */}
      {classInsights && (
        <Card className="border-l-4 border-l-purple-500 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Users className="w-5 h-5" />
              <span>Class Engagement Health</span>
              {getEngagementIcon(classInsights.overall_health)}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">
                  {classInsights.overall_health}%
                </div>
                <div className="text-sm text-gray-500">Overall Health</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-blue-600">
                  {classInsights.engagement_trend}
                </div>
                <div className="text-sm text-gray-500">Trend</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-green-600">
                  {classInsights.class_mood}
                </div>
                <div className="text-sm text-gray-500">Class Mood</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Insights & Recommendations */}
      {classInsights && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Key Insights */}
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Lightbulb className="w-5 h-5 text-yellow-500" />
                <span>AI Insights</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {classInsights.key_insights.map((insight, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                    <p className="text-sm">{insight}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Teaching Recommendations */}
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="w-5 h-5 text-blue-500" />
                <span>Smart Recommendations</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {classInsights.teaching_recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mt-2"></div>
                    <p className="text-sm">{rec}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Student Attention Alerts */}
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <span>Student Attention Alerts</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {studentAlerts.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <CheckCircle className="w-12 h-12 mx-auto mb-2 text-green-500" />
              <p>All students are engaged! 🎉</p>
            </div>
          ) : (
            <div className="space-y-4">
              {studentAlerts.map((alert, index) => (
                <Alert key={index} className="border-l-4 border-l-red-500">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="flex items-center justify-between">
                    <div>
                      <strong>Student #{alert.student_id}</strong>
                      <p className="text-sm mt-1">{alert.explanation}</p>
                      <Badge variant={getRiskBadgeColor(alert.risk_level)} className="mt-2">
                        {alert.risk_level.toUpperCase()} RISK
                      </Badge>
                    </div>
                    <Button 
                      size="sm" 
                      onClick={() => generateAdaptiveQuiz("current topic", "medium", alert.attention_score)}
                      className="ml-4"
                    >
                      <Gamepad2 className="w-4 h-4 mr-1" />
                      Generate Quiz
                    </Button>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Adaptive Quiz Modal */}
      {showQuiz && adaptiveQuiz && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-2xl mx-4">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Gamepad2 className="w-5 h-5 text-purple-500" />
                <span>Adaptive Engagement Quiz</span>
                <Badge variant="outline">AI Generated</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">{adaptiveQuiz.question}</h3>
                <div className="grid grid-cols-1 gap-2">
                  {adaptiveQuiz.options.map((option, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      className="justify-start h-auto p-4"
                      onClick={() => {
                        // Handle answer selection
                        console.log('Selected:', option);
                      }}
                    >
                      {String.fromCharCode(65 + index)}. {option}
                    </Button>
                  ))}
                </div>
                <div className="flex items-center justify-between pt-4 border-t">
                  <div className="text-sm text-gray-500">
                    <Clock className="w-4 h-4 inline mr-1" />
                    {adaptiveQuiz.estimated_time}
                  </div>
                  <div className="space-x-2">
                    <Button variant="outline" onClick={() => setShowQuiz(false)}>
                      Skip
                    </Button>
                    <Button onClick={() => setShowQuiz(false)}>
                      Submit
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Quick Action Panel */}
      {classInsights && (
        <Card className="shadow-lg border-l-4 border-l-green-500">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-green-500" />
              <span>Recommended Action</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold">{classInsights.recommended_activity}</p>
                <p className="text-sm text-gray-500">
                  Optimal timing: {classInsights.optimal_break_time}
                </p>
              </div>
              <Button className="bg-green-500 hover:bg-green-600">
                <MessageSquare className="w-4 h-4 mr-2" />
                Implement Now
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default GeminiAIDashboard;
