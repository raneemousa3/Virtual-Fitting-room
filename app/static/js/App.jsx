import React, { useState } from 'react';
import UploadArea from './components/UploadArea';

const App = () => {
  const [measurements, setMeasurements] = useState(null);
  const [height, setHeight] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    // This function will be called when an image is uploaded
    // It will be implemented to send the image to the backend
    console.log('Image uploaded:', file);
  };

  const handleHeightSubmit = async (e) => {
    e.preventDefault();
    if (!height) {
      setError('Please enter your height');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Here you would implement the API call to your backend
      // For now, we'll just simulate a delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock measurements for demonstration
      setMeasurements({
        chest: 95,
        waist: 80,
        hips: 100,
        inseam: 82,
        sleeve: 65
      });
    } catch (err) {
      setError('Failed to calculate measurements. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      <div className="container mx-auto px-4 py-12">
        <header className="text-center mb-16">
          <h1 className="font-serif text-5xl text-gray-900 mb-4">Counting On You</h1>
          <p className="text-gray-600 font-sans text-xl">Virtual Fitting Room</p>
        </header>

        <main className="max-w-4xl mx-auto">
          <UploadArea onImageUpload={handleImageUpload} />
          
          {measurements === null && (
            <form onSubmit={handleHeightSubmit} className="mt-12 bg-white p-8 rounded-lg shadow-sm border border-gray-100">
              <h2 className="font-serif text-3xl text-gray-800 mb-6 text-center">Your Measurements</h2>
              
              <div className="mb-6">
                <label htmlFor="height" className="block text-gray-700 font-sans mb-2">Height in Centimeters</label>
                <input 
                  type="number" 
                  id="height" 
                  value={height}
                  onChange={(e) => setHeight(e.target.value)}
                  min="100" 
                  max="250" 
                  step="0.1" 
                  placeholder="Enter your height"
                  className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#C8A165] focus:border-transparent"
                />
              </div>
              
              {error && <p className="text-red-500 mb-4">{error}</p>}
              
              <button 
                type="submit" 
                disabled={isLoading}
                className="w-full bg-black text-white py-3 px-6 rounded-md font-serif text-lg hover:bg-gray-800 transition-colors duration-300 disabled:opacity-50"
              >
                {isLoading ? 'Calculating...' : 'Calculate Measurements'}
              </button>
            </form>
          )}
          
          {measurements && (
            <div className="mt-12 bg-white p-8 rounded-lg shadow-sm border border-gray-100">
              <h2 className="font-serif text-3xl text-gray-800 mb-8 text-center">Your Body Measurements</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(measurements).map(([key, value]) => (
                  <div key={key} className="bg-gray-50 p-6 rounded-lg border border-gray-100 hover:border-[#C8A165] transition-colors duration-300">
                    <p className="text-gray-500 font-sans text-sm uppercase tracking-wider mb-2">{key}</p>
                    <p className="font-serif text-3xl text-gray-800">{value} cm</p>
                  </div>
                ))}
              </div>
              
              <button 
                onClick={() => setMeasurements(null)}
                className="mt-8 mx-auto block text-[#C8A165] font-serif border-b border-[#C8A165] pb-1 hover:text-black hover:border-black transition-colors duration-300"
              >
                Start Over
              </button>
            </div>
          )}
        </main>
        
        {isLoading && (
          <div className="fixed inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="text-center">
              <div className="inline-block h-12 w-12 border-4 border-[#C8A165] border-t-transparent rounded-full animate-spin mb-4"></div>
              <p className="font-serif text-xl text-gray-800">Processing your measurements...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App; 