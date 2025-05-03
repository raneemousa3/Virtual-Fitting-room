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
        formData.append('file', currentFile);
        formData.append('height', height.toString());

        try {
            loadingOverlay.style.display = 'flex';
            const response = await fetch('/api/v1/measurements/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to process image');
            }

            const data = await response.json();
            console.log('Response data:', data);
            displayMeasurements(data.measurements);
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

    function displayMeasurements(measurements) {
        console.log('Displaying measurements:', measurements);
        measurementsGrid.innerHTML = '';
        
        const measurementTypes = {
            chest_cm: 'Chest',
            waist_cm: 'Waist',
            hips_cm: 'Hips',
            sleeve_cm: 'Sleeve Length',
            inseam_cm: 'Inseam'
        };

        let hasMeasurements = false;
        for (const [key, label] of Object.entries(measurementTypes)) {
            console.log(`Checking measurement: ${key}, value:`, measurements[key]);
            if (measurements[key]) {
                hasMeasurements = true;
                const card = document.createElement('div');
                card.className = 'measurement-card';
                card.innerHTML = `
                    <h3>${label}</h3>
                    <p>${measurements[key].toFixed(1)} cm</p>
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

        resultsSection.style.display = 'block';
    }
}); 