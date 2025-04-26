import React from 'react';

const MeasurementsDisplay = ({ measurements }) => {
  if (!measurements) return null;

  const formatMeasurement = (value) => {
    return `${value.toFixed(1)} cm`;
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-serif text-gray-800 mb-6 text-center">
        Your Measurements
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="flex justify-between items-center p-4 bg-ivory/30 rounded-lg">
            <span className="font-serif text-gray-700">Chest</span>
            <span className="font-sans font-medium">{formatMeasurement(measurements.chest)}</span>
          </div>
          <div className="flex justify-between items-center p-4 bg-ivory/30 rounded-lg">
            <span className="font-serif text-gray-700">Waist</span>
            <span className="font-sans font-medium">{formatMeasurement(measurements.waist)}</span>
          </div>
          <div className="flex justify-between items-center p-4 bg-ivory/30 rounded-lg">
            <span className="font-serif text-gray-700">Hips</span>
            <span className="font-sans font-medium">{formatMeasurement(measurements.hips)}</span>
          </div>
        </div>
        <div className="space-y-4">
          <div className="flex justify-between items-center p-4 bg-ivory/30 rounded-lg">
            <span className="font-serif text-gray-700">Shoulder Width</span>
            <span className="font-sans font-medium">{formatMeasurement(measurements.shoulder_width)}</span>
          </div>
          <div className="flex justify-between items-center p-4 bg-ivory/30 rounded-lg">
            <span className="font-serif text-gray-700">Arm Length</span>
            <span className="font-sans font-medium">{formatMeasurement(measurements.arm_length)}</span>
          </div>
          <div className="flex justify-between items-center p-4 bg-ivory/30 rounded-lg">
            <span className="font-serif text-gray-700">Inseam</span>
            <span className="font-sans font-medium">{formatMeasurement(measurements.inseam)}</span>
          </div>
        </div>
      </div>
      {measurements.fit_rating && (
        <div className="mt-6 p-4 bg-gold/10 rounded-lg text-center">
          <p className="font-serif text-gold">
            Fit Rating: {measurements.fit_rating}
          </p>
        </div>
      )}
    </div>
  );
};

export default MeasurementsDisplay; 