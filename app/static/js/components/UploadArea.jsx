import React, { useState, useRef } from 'react';

const UploadArea = ({ onImageUpload }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Check file type and size
    if (!file.type.match('image/(jpeg|png)')) {
      alert('Please upload a JPG or PNG image');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB
      alert('File size must be less than 10MB');
      return;
    }

    // Create preview URL
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewUrl(e.target.result);
      onImageUpload(file);
    };
    reader.readAsDataURL(file);
  };

  const handleBrowseClick = () => {
    fileInputRef.current.click();
  };

  const handleRemoveImage = () => {
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      {!previewUrl ? (
        <div 
          className={`border border-[#C8A165] rounded-lg p-12 flex flex-col items-center justify-center transition-all duration-300 ${
            isDragging ? 'bg-black/5' : 'bg-white'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="text-[#C8A165] mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
            </svg>
          </div>
          
          <h3 className="font-serif text-2xl text-gray-800 mb-4">Upload Image</h3>
          
          <button 
            onClick={handleBrowseClick}
            className="text-[#C8A165] font-serif text-lg border-b border-[#C8A165] pb-1 mb-6 hover:text-black hover:border-black transition-colors duration-300"
          >
            Browse files
          </button>
          
          <button 
            onClick={handleBrowseClick}
            className="bg-white border border-black text-black px-6 py-3 rounded-md flex items-center gap-2 font-serif hover:bg-black hover:text-[#F5F5F0] transition-colors duration-300"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#C8A165]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Upload Image
          </button>
          
          <p className="text-gray-500 text-sm mt-6 font-sans">JPG, PNG • Max 10MB</p>
          
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileInputChange}
            accept="image/jpeg,image/png"
            className="hidden"
          />
        </div>
      ) : (
        <div className="relative border border-[#C8A165] rounded-lg overflow-hidden">
          <img 
            src={previewUrl} 
            alt="Preview" 
            className="w-full h-auto max-h-[500px] object-contain"
          />
          <button 
            onClick={handleRemoveImage}
            className="absolute top-4 right-4 bg-white/90 hover:bg-white text-gray-800 rounded-full p-2 transition-colors duration-300"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
};

export default UploadArea; 