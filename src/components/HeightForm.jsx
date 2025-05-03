import React, { useState } from 'react';

function HeightForm({ onSubmit }) {
  const [height, setHeight] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!height) {
      setError('Please enter your height');
      return;
    }

    const heightNum = parseFloat(height);
    if (isNaN(heightNum) || heightNum < 100 || heightNum > 250) {
      setError('Please enter a valid height between 100 and 250 cm');
      return;
    }

    onSubmit(heightNum);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <h2 className="text-2xl font-serif text-gray-900 mb-4 text-center">
        Enter Your Height
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label 
            htmlFor="height" 
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Height (cm)
          </label>
          <input
            type="number"
            id="height"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
            min="100"
            max="250"
            step="0.1"
            placeholder="Enter your height in centimeters"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        {error && (
          <p className="text-red-600 text-sm">{error}</p>
        )}

        <button
          type="submit"
          className="w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-colors duration-200"
        >
          Calculate Measurements
        </button>
      </form>
    </div>
  );
}

export default HeightForm; 