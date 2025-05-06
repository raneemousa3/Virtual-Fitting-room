document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('uploadBox');
    const imageInput = document.getElementById('imageInput');
    const previewBox = document.getElementById('previewBox');
    const imagePreview = document.getElementById('imagePreview');
    const removeImage = document.getElementById('removeImage');
    const measurementsForm = document.getElementById('measurementsForm');
    const heightInput = document.getElementById('height');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    const measurementsGrid = document.getElementById('measurementsGrid');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const browseLink = document.querySelector('.browse-link');

    let currentFile = null;

    // Handle drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--primary-color)';
        uploadBox.style.backgroundColor = 'rgba(99, 102, 241, 0.05)';
    });

    uploadBox.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--border-color)';
        uploadBox.style.backgroundColor = 'transparent';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--border-color)';
        uploadBox.style.backgroundColor = 'transparent';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            currentFile = file;
            handleImageUpload(file);
        }
    });

    // Handle click to upload
    uploadBox.addEventListener('click', () => {
        imageInput.click();
    });

    browseLink.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            currentFile = file;
            handleImageUpload(file);
        }
    });

    // Handle image removal
    removeImage.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Handle form submission
    submitBtn.addEventListener('click', async () => {
        const height = parseFloat(heightInput.value);
        if (!height || isNaN(height) || height < 100 || height > 250) {
            alert('Please enter a valid height between 100 and 250 cm');
            return;
        }

        if (!currentFile) {
            alert('Please upload an image first');
            return;
        }

        const formData = new FormData();
        formData.append('mask_image', currentFile);
        formData.append('mask_left_image', currentFile);  // For now, using same image for both views
        formData.append('camera_info', JSON.stringify({
            focal_length: 50.0,
            sensor_height: 24.0,
            image_height: 256.0
        }));

        try {
            loadingOverlay.style.display = 'flex';
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to process image');
            }

            const data = await response.json();
            console.log('Response data:', data);
            displayMeasurements(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing image: ' + error.message);
        } finally {
            loadingOverlay.style.display = 'none';
        }
    });

    function handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadBox.style.display = 'none';
            previewBox.style.display = 'block';
            measurementsForm.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        currentFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        uploadBox.style.display = 'block';
        previewBox.style.display = 'none';
        measurementsForm.style.display = 'none';
        resultsSection.style.display = 'none';
        heightInput.value = '';
    }

    function displayMeasurements(data) {
        console.log('Displaying measurements:', data);
        measurementsGrid.innerHTML = '';
        
        const measurementTypes = {
            'height': 'Height',
            'shoulder-to-crotch': 'Shoulder to Crotch',
            'waist': 'Waist',
            'chest': 'Chest',
            'shoulder-breadth': 'Shoulder Breadth',
            'hip': 'Hip',
            'ankle': 'Ankle',
            'arm-length': 'Arm Length',
            'bicep': 'Bicep',
            'calf': 'Calf',
            'forearm': 'Forearm',
            'leg-length': 'Leg Length',
            'thigh': 'Thigh',
            'wrist': 'Wrist'
        };

        let hasMeasurements = false;
        for (const [key, label] of Object.entries(measurementTypes)) {
            if (data.measurements[key]) {
                hasMeasurements = true;
                const confidence = data.confidence_scores[key] || 0;
                const confidenceClass = confidence > 0.8 ? 'high' : confidence > 0.6 ? 'medium' : 'low';
                
                const card = document.createElement('div');
                card.className = 'measurement-card';
                card.innerHTML = `
                    <h3>${label}</h3>
                    <p class="measurement-value">${data.measurements[key].toFixed(1)} cm</p>
                    <div class="confidence-indicator">
                        <div class="confidence-bar ${confidenceClass}" style="width: ${confidence * 100}%"></div>
                        <span class="confidence-text">${(confidence * 100).toFixed(0)}% confidence</span>
                    </div>
                `;
                measurementsGrid.appendChild(card);
            }
        }

        if (!hasMeasurements) {
            console.log('No measurements found in data');
            const errorCard = document.createElement('div');
            errorCard.className = 'measurement-card error';
            errorCard.innerHTML = `
                <h3>No Measurements Available</h3>
                <p>Please try uploading a different image or ensure your pose is clearly visible.</p>
            `;
            measurementsGrid.appendChild(errorCard);
        }

        // Add body type summary if available
        if (data.body_summary) {
            const summaryCard = document.createElement('div');
            summaryCard.className = 'summary-card';
            summaryCard.innerHTML = `
                <h3>Body Type Summary</h3>
                <p><strong>Body Type:</strong> ${data.body_summary.body_type}</p>
                <p><strong>Size Category:</strong> ${data.body_summary.size_category}</p>
                <p><strong>Height Category:</strong> ${data.body_summary.height_category}</p>
                <p><strong>Waist-to-Hip Ratio:</strong> ${data.body_summary.waist_to_hip_ratio}</p>
            `;
            measurementsGrid.appendChild(summaryCard);
        }

        resultsSection.style.display = 'block';
    }
}); 