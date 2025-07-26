import { AIDashboard } from './components/dashboard/AIDashboard';
import { AIProvider } from './stores/aiStore';
import './styles/globals.css';
function App() {
  return (
    <AIProvider>
      <AIDashboard />
    </AIProvider>
  );
}
export default App;