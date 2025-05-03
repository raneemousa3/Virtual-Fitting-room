import React, { useState } from 'react';
import UploadArea from './components/UploadArea';
import HeightForm from './components/HeightForm';
import MeasurementsDisplay from './components/MeasurementsDisplay';

function App() {
  const [image, setImage] = useState(null);
  const [measurements, setMeasurements] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (file) => {
    setImage(file);
    setMeasurements(null);
    setError(null);
  };

  const handleHeightSubmit = async (height) => {
    if (!image) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', image);
      formData.append('height', height);

      const response = await fetch('/api/v1/measurements/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process measurements');
      }

      const data = await response.json();
      setMeasurements(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-serif text-gray-900 mb-4">
            Virtual Fitting Room
          </h1>
          <p className="text-lg text-gray-600">
            Upload a photo and get your precise measurements
          </p>
        </header>

        <main className="space-y-8">
          <UploadArea onImageUpload={handleImageUpload} />
          
          {image && (
            <HeightForm onSubmit={handleHeightSubmit} />
          )}

          {loading && (
            <div className="flex justify-center items-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gold"></div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-center">
              {error}
            </div>
          )}

          {measurements && !loading && (
            <MeasurementsDisplay measurements={measurements} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App; 